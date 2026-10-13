//! Distributed Edge Swarms (`sofuu swarm`)
//!
//! Sequential Layer-Wise Activation Pipelining across distributed edge swarm nodes
//! (Raspberry Pi 5 clusters, mixed PCs, Mac laptops, and mobile devices over LAN).
//!
//! A large model (e.g. 14B–32B Q4_K or 70B MoE) is split across multiple edge nodes:
//! - Node 1 (Coordinator/Head): Tokenizer, Token Embedding, Layers 0..L1
//! - Node 2 (Worker):          Layers L1..L2
//! - ...
//! - Node N (Tail):            Layers LN-1..Ltotal, Output Norm, LM Head, and Sampler
//!
//! Across LAN, nodes transfer compact 4 KB–16 KB activation vectors per token over
//! low-latency TCP streams (taking <0.1ms over Gigabit LAN), enabling cheap devices
//! to pool RAM and execute models that would otherwise not fit on a single device.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::{GenerateOptions, InferEngine};
use super::sampler::Sampler;

/// Magic bytes framing every packet: b"SFSW" (Sofuu Swarm)
pub const SWARM_MAGIC: [u8; 4] = *b"SFSW";

/// Protocol message types
pub const MSG_HANDSHAKE: u8 = 0x01;
pub const MSG_HANDSHAKE_ACK: u8 = 0x02;
pub const MSG_FORWARD: u8 = 0x03;
pub const MSG_REPLY: u8 = 0x04;
pub const MSG_RESET: u8 = 0x05;
pub const MSG_PING: u8 = 0x06;
pub const MSG_PONG: u8 = 0x07;
pub const MSG_ERROR: u8 = 0xFF;

/// Default UDP discovery beacon port
pub const DEFAULT_DISCOVERY_PORT: u16 = 9770;
/// Default TCP swarm communication port
pub const DEFAULT_SWARM_PORT: u16 = 9001;

// ── Framing & Wire Protocol ──────────────────────────────────────────────────

pub struct SwarmProto;

impl SwarmProto {
    /// Write a framed binary packet to a TCP stream:
    /// [Magic: 4 bytes][MsgType: 1 byte][PayloadLen: 4 bytes LE][Payload: PayloadLen bytes]
    pub fn write_packet(stream: &mut TcpStream, msg_type: u8, payload: &[u8]) -> std::io::Result<()> {
        let mut header = [0u8; 9];
        header[0..4].copy_from_slice(&SWARM_MAGIC);
        header[4] = msg_type;
        header[5..9].copy_from_slice(&(payload.len() as u32).to_le_bytes());

        stream.write_all(&header)?;
        if !payload.is_empty() {
            stream.write_all(payload)?;
        }
        stream.flush()
    }

    /// Read a framed binary packet from a TCP stream
    pub fn read_packet(stream: &mut TcpStream) -> std::io::Result<(u8, Vec<u8>)> {
        let mut header = [0u8; 9];
        stream.read_exact(&mut header)?;

        if header[0..4] != SWARM_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid swarm magic: {:?}", &header[0..4]),
            ));
        }

        let msg_type = header[4];
        let payload_len = u32::from_le_bytes([header[5], header[6], header[7], header[8]]) as usize;

        let mut payload = vec![0u8; payload_len];
        if payload_len > 0 {
            stream.read_exact(&mut payload)?;
        }

        Ok((msg_type, payload))
    }
}

// ── Packet Serialization Helpers ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct HandshakePayload {
    pub node_id: usize,
    pub total_nodes: usize,
    pub start_layer: usize,
    pub end_layer: usize,
    pub total_layers: usize,
    pub dim: usize,
    pub model_name: String,
}

impl HandshakePayload {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32 + self.model_name.len());
        buf.extend_from_slice(&(self.node_id as u32).to_le_bytes());
        buf.extend_from_slice(&(self.total_nodes as u32).to_le_bytes());
        buf.extend_from_slice(&(self.start_layer as u32).to_le_bytes());
        buf.extend_from_slice(&(self.end_layer as u32).to_le_bytes());
        buf.extend_from_slice(&(self.total_layers as u32).to_le_bytes());
        buf.extend_from_slice(&(self.dim as u32).to_le_bytes());
        let name_bytes = self.model_name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf
    }

    pub fn decode(buf: &[u8]) -> Result<Self, String> {
        if buf.len() < 28 {
            return Err("Handshake payload too short".to_string());
        }
        let node_id = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        let total_nodes = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
        let start_layer = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        let end_layer = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
        let total_layers = u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(buf[20..24].try_into().unwrap()) as usize;
        let name_len = u32::from_le_bytes(buf[24..28].try_into().unwrap()) as usize;
        if buf.len() < 28 + name_len {
            return Err("Handshake model name truncated".to_string());
        }
        let model_name = String::from_utf8_lossy(&buf[28..28 + name_len]).to_string();

        Ok(Self {
            node_id,
            total_nodes,
            start_layer,
            end_layer,
            total_layers,
            dim,
            model_name,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ForwardPayload {
    pub seq_id: u64,
    pub pos: usize,
    pub layer_start: usize,
    pub layer_end: usize,
    pub dim: usize,
    pub activations: Vec<f32>,
}

impl ForwardPayload {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(24 + self.activations.len() * 4);
        buf.extend_from_slice(&self.seq_id.to_le_bytes());
        buf.extend_from_slice(&(self.pos as u32).to_le_bytes());
        buf.extend_from_slice(&(self.layer_start as u32).to_le_bytes());
        buf.extend_from_slice(&(self.layer_end as u32).to_le_bytes());
        buf.extend_from_slice(&(self.dim as u32).to_le_bytes());

        // Zero-copy float slice cast to byte stream
        let float_bytes = unsafe {
            std::slice::from_raw_parts(
                self.activations.as_ptr() as *const u8,
                self.activations.len() * std::mem::size_of::<f32>(),
            )
        };
        buf.extend_from_slice(float_bytes);
        buf
    }

    pub fn decode(buf: &[u8]) -> Result<Self, String> {
        if buf.len() < 24 {
            return Err("Forward payload too short".to_string());
        }
        let seq_id = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let pos = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        let layer_start = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
        let layer_end = u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(buf[20..24].try_into().unwrap()) as usize;

        let expected_bytes = dim * 4;
        if buf.len() < 24 + expected_bytes {
            return Err(format!(
                "Forward activation buffer truncated: got {}, expected {}",
                buf.len() - 24,
                expected_bytes
            ));
        }

        let mut activations = vec![0.0f32; dim];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf[24..24 + expected_bytes].as_ptr(),
                activations.as_mut_ptr() as *mut u8,
                expected_bytes,
            );
        }

        Ok(Self {
            seq_id,
            pos,
            layer_start,
            layer_end,
            dim,
            activations,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ReplyPayload {
    pub seq_id: u64,
    pub pos: usize,
    pub token_id: Option<u32>,
    pub activations: Vec<f32>,
}

impl ReplyPayload {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(17 + self.activations.len() * 4);
        buf.extend_from_slice(&self.seq_id.to_le_bytes());
        buf.extend_from_slice(&(self.pos as u32).to_le_bytes());
        if let Some(tok) = self.token_id {
            buf.push(1);
            buf.extend_from_slice(&tok.to_le_bytes());
        } else {
            buf.push(0);
            buf.extend_from_slice(&0u32.to_le_bytes());
        }
        buf.extend_from_slice(&(self.activations.len() as u32).to_le_bytes());
        let float_bytes = unsafe {
            std::slice::from_raw_parts(
                self.activations.as_ptr() as *const u8,
                self.activations.len() * std::mem::size_of::<f32>(),
            )
        };
        buf.extend_from_slice(float_bytes);
        buf
    }

    pub fn decode(buf: &[u8]) -> Result<Self, String> {
        if buf.len() < 21 {
            return Err("Reply payload too short".to_string());
        }
        let seq_id = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let pos = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        let has_tok = buf[12] == 1;
        let token_val = u32::from_le_bytes(buf[13..17].try_into().unwrap());
        let token_id = if has_tok { Some(token_val) } else { None };
        let act_len = u32::from_le_bytes(buf[17..21].try_into().unwrap()) as usize;

        let expected_bytes = act_len * 4;
        let mut activations = vec![0.0f32; act_len];
        if act_len > 0 {
            if buf.len() < 21 + expected_bytes {
                return Err("Reply activations truncated".to_string());
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buf[21..21 + expected_bytes].as_ptr(),
                    activations.as_mut_ptr() as *mut u8,
                    expected_bytes,
                );
            }
        }

        Ok(Self {
            seq_id,
            pos,
            token_id,
            activations,
        })
    }
}

// ── Partitioning & Topology ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerPartition {
    pub node_id: usize,
    pub total_nodes: usize,
    pub start_layer: usize,
    pub end_layer: usize,
    pub total_layers: usize,
}

impl LayerPartition {
    pub fn is_head(&self) -> bool {
        self.start_layer == 0
    }

    pub fn is_tail(&self) -> bool {
        self.end_layer >= self.total_layers
    }

    pub fn num_layers(&self) -> usize {
        self.end_layer.saturating_sub(self.start_layer)
    }

    /// Parse a range string such as "0..16", "16..32", or "0-15"
    pub fn parse_range(range_str: &str, total_layers: usize) -> Result<(usize, usize), String> {
        let clean = range_str.trim();
        if let Some((s, e)) = clean.split_once("..") {
            let start = s.trim().parse::<usize>().map_err(|e| format!("Invalid start layer: {e}"))?;
            let end = e.trim().parse::<usize>().map_err(|e| format!("Invalid end layer: {e}"))?;
            if start >= end || end > total_layers {
                return Err(format!("Layer range {}..{} out of bounds for total {}", start, end, total_layers));
            }
            Ok((start, end))
        } else if let Some((s, e)) = clean.split_once('-') {
            let start = s.trim().parse::<usize>().map_err(|e| format!("Invalid start layer: {e}"))?;
            let end_inclusive = e.trim().parse::<usize>().map_err(|e| format!("Invalid end layer: {e}"))?;
            let end = end_inclusive + 1;
            if start >= end || end > total_layers {
                return Err(format!("Layer range {}-{} out of bounds for total {}", start, end_inclusive, total_layers));
            }
            Ok((start, end))
        } else {
            Err(format!("Invalid layer range format '{}' (expected '0..16' or '0-15')", range_str))
        }
    }

    /// Automatically and evenly partition `total_layers` across `num_nodes`
    pub fn auto_partition(total_layers: usize, num_nodes: usize) -> Vec<(usize, usize)> {
        if num_nodes == 0 || total_layers == 0 {
            return Vec::new();
        }
        let base = total_layers / num_nodes;
        let remainder = total_layers % num_nodes;

        let mut partitions = Vec::with_capacity(num_nodes);
        let mut current = 0;
        for i in 0..num_nodes {
            let count = base + if i < remainder { 1 } else { 0 };
            partitions.push((current, current + count));
            current += count;
        }
        partitions
    }
}

// ── UDP Auto-Discovery Beacon ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DiscoveredPeer {
    pub cluster_id: String,
    pub role: String,
    pub addr: String,
    pub layers: String,
    pub model: String,
}

pub struct SwarmDiscovery;

impl SwarmDiscovery {
    /// Broadcast a discovery beacon over UDP LAN
    pub fn broadcast_beacon(
        cluster_id: &str,
        role: &str,
        service_port: u16,
        layers: &str,
        model: &str,
        running: Arc<AtomicBool>,
    ) {
        let socket = match UdpSocket::bind("0.0.0.0:0") {
            Ok(s) => s,
            Err(_) => return,
        };
        let _ = socket.set_broadcast(true);

        let msg = format!(
            "{{\"magic\":\"sofuu-swarm\",\"cluster\":\"{}\",\"role\":\"{}\",\"port\":{},\"layers\":\"{}\",\"model\":\"{}\"}}",
            cluster_id, role, service_port, layers, model
        );

        let broadcast_addr = format!("255.255.255.255:{}", DEFAULT_DISCOVERY_PORT);
        std::thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                let _ = socket.send_to(msg.as_bytes(), &broadcast_addr);
                std::thread::sleep(Duration::from_millis(1000));
            }
        });
    }

    /// Listen for discovery beacons for `timeout` duration
    pub fn listen(timeout: Duration) -> Vec<DiscoveredPeer> {
        let socket = match UdpSocket::bind(format!("0.0.0.0:{}", DEFAULT_DISCOVERY_PORT)) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let _ = socket.set_read_timeout(Some(timeout));

        let start = Instant::now();
        let mut peers = Vec::new();
        let mut buf = [0u8; 1024];

        while start.elapsed() < timeout {
            if let Ok((len, src)) = socket.recv_from(&mut buf) {
                if let Ok(text) = std::str::from_utf8(&buf[..len]) {
                    if text.contains("\"magic\":\"sofuu-swarm\"") {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
                            let cluster = v.get("cluster").and_then(|s| s.as_str()).unwrap_or_default().to_string();
                            let role = v.get("role").and_then(|s| s.as_str()).unwrap_or_default().to_string();
                            let port = v.get("port").and_then(|p| p.as_u64()).unwrap_or(DEFAULT_SWARM_PORT as u64) as u16;
                            let layers = v.get("layers").and_then(|s| s.as_str()).unwrap_or_default().to_string();
                            let model = v.get("model").and_then(|s| s.as_str()).unwrap_or_default().to_string();

                            let peer_addr = format!("{}:{}", src.ip(), port);
                            if !peers.iter().any(|p: &DiscoveredPeer| p.addr == peer_addr) {
                                peers.push(DiscoveredPeer {
                                    cluster_id: cluster,
                                    role,
                                    addr: peer_addr,
                                    layers,
                                    model,
                                });
                            }
                        }
                    }
                }
            }
        }
        peers
    }
}

// ── Swarm Worker Daemon ──────────────────────────────────────────────────────

pub struct SwarmWorker {
    pub partition: LayerPartition,
    pub engine: InferEngine,
}

impl SwarmWorker {
    pub fn new(engine: InferEngine, partition: LayerPartition) -> Self {
        Self { partition, engine }
    }

    /// Run the worker TCP listener and serve activation forwarding requests
    pub fn serve(&mut self, listen_addr: &str, running: Option<Arc<AtomicBool>>) -> Result<(), String> {
        let listener = TcpListener::bind(listen_addr)
            .map_err(|e| format!("Failed to bind worker listener on {}: {}", listen_addr, e))?;

        println!(
            "\x1b[36m⚡ Sofuu Swarm Worker\x1b[0m listening on {} [layers {}..{}/{}]",
            listen_addr,
            self.partition.start_layer,
            self.partition.end_layer,
            self.partition.total_layers
        );

        let running_flag = running.unwrap_or_else(|| Arc::new(AtomicBool::new(true)));

        while running_flag.load(Ordering::Relaxed) {
            listener.set_nonblocking(true).ok();
            match listener.accept() {
                Ok((mut stream, peer_addr)) => {
                    stream.set_nonblocking(false).ok();
                    stream.set_nodelay(true).ok();
                    println!("\x1b[32m✓ Upstream connected from {}\x1b[0m", peer_addr);

                    if let Err(e) = self.handle_session(&mut stream, &running_flag) {
                        eprintln!("\x1b[33mWorker session ended:\x1b[0m {}", e);
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => {
                    eprintln!("Worker accept error: {}", e);
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }

        Ok(())
    }

    fn handle_session(&mut self, stream: &mut TcpStream, running: &Arc<AtomicBool>) -> Result<(), String> {
        let dim = self.engine.model.config.dim;
        let mut out_hidden = vec![0.0f32; dim];

        while running.load(Ordering::Relaxed) {
            let (msg_type, payload) = match SwarmProto::read_packet(stream) {
                Ok(res) => res,
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    return Ok(()); // Clean upstream close
                }
                Err(e) => return Err(format!("Socket read error: {}", e)),
            };

            match msg_type {
                MSG_HANDSHAKE => {
                    let hs = HandshakePayload::decode(&payload)?;
                    println!(
                        "  Handshake: node {}/{} running layers {}..{} for model '{}'",
                        hs.node_id, hs.total_nodes, hs.start_layer, hs.end_layer, hs.model_name
                    );
                    let ack = HandshakePayload {
                        node_id: self.partition.node_id,
                        total_nodes: self.partition.total_nodes,
                        start_layer: self.partition.start_layer,
                        end_layer: self.partition.end_layer,
                        total_layers: self.partition.total_layers,
                        dim,
                        model_name: hs.model_name,
                    };
                    SwarmProto::write_packet(stream, MSG_HANDSHAKE_ACK, &ack.encode())
                        .map_err(|e| format!("Failed to send handshake ack: {}", e))?;
                }
                MSG_FORWARD => {
                    let fwd = ForwardPayload::decode(&payload)?;
                    // Execute assigned transformer block range
                    self.engine.forward_layer_range(
                        fwd.pos,
                        self.partition.start_layer,
                        self.partition.end_layer,
                        &fwd.activations,
                        &mut out_hidden,
                    );

                    let reply = ReplyPayload {
                        seq_id: fwd.seq_id,
                        pos: fwd.pos,
                        token_id: None,
                        activations: out_hidden.clone(),
                    };
                    SwarmProto::write_packet(stream, MSG_REPLY, &reply.encode())
                        .map_err(|e| format!("Failed to write reply: {}", e))?;
                }
                MSG_RESET => {
                    self.engine.reset_cache();
                    SwarmProto::write_packet(stream, MSG_PONG, &[])
                        .map_err(|e| format!("Failed to ack reset: {}", e))?;
                }
                MSG_PING => {
                    SwarmProto::write_packet(stream, MSG_PONG, &[])
                        .map_err(|e| format!("Failed to pong: {}", e))?;
                }
                _ => {
                    eprintln!("Unknown message type: 0x{:02x}", msg_type);
                }
            }
        }
        Ok(())
    }
}

// ── Swarm Leader / Coordinator ───────────────────────────────────────────────

pub struct SwarmLeader {
    pub partition: LayerPartition,
    pub engine: InferEngine,
    pub worker_streams: Vec<TcpStream>,
}

impl SwarmLeader {
    pub fn new(engine: InferEngine, partition: LayerPartition) -> Self {
        Self {
            partition,
            engine,
            worker_streams: Vec::new(),
        }
    }

    /// Connect to worker peers in the pipeline
    pub fn connect_workers(&mut self, worker_addrs: &[String]) -> Result<(), String> {
        let dim = self.engine.model.config.dim;
        let total_layers = self.partition.total_layers;

        for (idx, addr) in worker_addrs.iter().enumerate() {
            println!("\x1b[36mConnecting to swarm worker [{}/{}]: {}\x1b[0m", idx + 1, worker_addrs.len(), addr);
            let mut stream = TcpStream::connect(addr)
                .map_err(|e| format!("Failed to connect to worker at {}: {}", addr, e))?;
            stream.set_nodelay(true).ok();

            // Send handshake
            let hs = HandshakePayload {
                node_id: 0,
                total_nodes: worker_addrs.len() + 1,
                start_layer: self.partition.start_layer,
                end_layer: self.partition.end_layer,
                total_layers,
                dim,
                model_name: "sofuu-swarm-cluster".to_string(),
            };
            SwarmProto::write_packet(&mut stream, MSG_HANDSHAKE, &hs.encode())
                .map_err(|e| format!("Handshake write error to {}: {}", addr, e))?;

            // Await ack
            let (mtype, reply_buf) = SwarmProto::read_packet(&mut stream)
                .map_err(|e| format!("Handshake ack read error from {}: {}", addr, e))?;

            if mtype != MSG_HANDSHAKE_ACK {
                return Err(format!("Worker {} responded with unexpected message 0x{:02x}", addr, mtype));
            }

            let ack = HandshakePayload::decode(&reply_buf)?;
            println!(
                "\x1b[32m✓ Worker {} connected:\x1b[0m node {} running layers {}..{}/{}",
                addr, ack.node_id, ack.start_layer, ack.end_layer, ack.total_layers
            );

            self.worker_streams.push(stream);
        }

        Ok(())
    }

    /// Stream inference over the distributed swarm cluster
    pub fn stream<F>(&mut self, prompt: &str, options: GenerateOptions, mut on_token: F) -> Result<String, String>
    where
        F: FnMut(&str) -> bool,
    {
        if self.worker_streams.is_empty() {
            return Err("No swarm workers connected. Run with --connect or join nodes first.".to_string());
        }

        let dim = self.engine.model.config.dim;
        let vocab_size = self.engine.model.config.vocab_size;

        // Reset all nodes in the cluster
        self.engine.reset_cache();
        for stream in &mut self.worker_streams {
            let _ = SwarmProto::write_packet(stream, MSG_RESET, &[]);
            let _ = SwarmProto::read_packet(stream);
        }

        let prompt_tokens = self.engine.tokenizer.encode(prompt);
        if prompt_tokens.is_empty() {
            return Err("Empty prompt tokens".to_string());
        }

        use super::sampler::SamplerConfig;
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: options.temperature,
            top_p: options.top_p,
            top_k: options.top_k,
            repetition_penalty: options.repetition_penalty,
        });

        let mut output = String::new();
        let mut hidden = vec![0.0f32; dim];
        let mut worker_hidden = vec![0.0f32; dim];
        let mut logits = vec![0.0f32; vocab_size];

        let mut cur_pos = 0;
        let seq_id = 1u64;

        // ── 1. Prefill Phase across the Swarm ──
        let t_prefill_start = Instant::now();
        for &tok in &prompt_tokens {
            // Coordinator embeds token and runs head layers
            self.engine.forward_embed(tok, &mut hidden);
            self.engine.forward_layer_range(
                cur_pos,
                self.partition.start_layer,
                self.partition.end_layer,
                &hidden.clone(),
                &mut worker_hidden,
            );

            // Pipeline activations across worker nodes
            for stream in &mut self.worker_streams {
                let fwd = ForwardPayload {
                    seq_id,
                    pos: cur_pos,
                    layer_start: self.partition.end_layer,
                    layer_end: self.partition.total_layers,
                    dim,
                    activations: worker_hidden.clone(),
                };
                SwarmProto::write_packet(stream, MSG_FORWARD, &fwd.encode())
                    .map_err(|e| format!("Prefill forward error: {}", e))?;

                let (_, reply_buf) = SwarmProto::read_packet(stream)
                    .map_err(|e| format!("Prefill reply read error: {}", e))?;
                let rep = ReplyPayload::decode(&reply_buf)?;
                worker_hidden = rep.activations;
            }

            cur_pos += 1;
        }

        let prefill_dur = t_prefill_start.elapsed();
        let _ = prefill_dur; // Available for diagnostics

        // Compute logits for the last prompt token
        self.engine.forward_head(&worker_hidden, &mut logits);
        let mut next_token = sampler.sample(&mut logits, &prompt_tokens);

        let mut _generated_count = 0usize;
        let mut generated_tokens = Vec::new();

        // ── 2. Autoregressive Decode Phase ──
        for _ in 0..options.max_tokens {
            let is_eos = self.engine.tokenizer.eos_token_id == Some(next_token);
            let piece = self.engine.tokenizer.decode_piece(next_token);
            let raw = self.engine.tokenizer.decode_token(next_token);

            let is_stop = is_eos
                || raw == Some("<|im_end|>")
                || raw == Some("<|endoftext|>")
                || raw == Some("<|eot_id|>")
                || options.stop_tokens.iter().any(|s| raw == Some(s.as_str()))
                || options.stop_tokens.iter().any(|s| piece.contains(s.as_str()));

            if is_stop {
                break;
            }

            output.push_str(&piece);
            generated_tokens.push(next_token);
            _generated_count += 1;

            if !on_token(&piece) {
                break;
            }

            // Step token forward across the swarm
            self.engine.forward_embed(next_token, &mut hidden);
            self.engine.forward_layer_range(
                cur_pos,
                self.partition.start_layer,
                self.partition.end_layer,
                &hidden.clone(),
                &mut worker_hidden,
            );

            // Send through worker pipeline
            for stream in &mut self.worker_streams {
                let fwd = ForwardPayload {
                    seq_id,
                    pos: cur_pos,
                    layer_start: self.partition.end_layer,
                    layer_end: self.partition.total_layers,
                    dim,
                    activations: worker_hidden.clone(),
                };
                SwarmProto::write_packet(stream, MSG_FORWARD, &fwd.encode())
                    .map_err(|e| format!("Decode forward error: {}", e))?;

                let (_, reply_buf) = SwarmProto::read_packet(stream)
                    .map_err(|e| format!("Decode reply read error: {}", e))?;
                let rep = ReplyPayload::decode(&reply_buf)?;
                worker_hidden = rep.activations;
            }

            // Coordinator applies final head and samples
            self.engine.forward_head(&worker_hidden, &mut logits);

            let mut history = prompt_tokens.clone();
            history.extend_from_slice(&generated_tokens);
            next_token = sampler.sample(&mut logits, &history);
            cur_pos += 1;
        }

        Ok(output)
    }
}

// ── Unit Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_partition_auto() {
        // 32 layers across 2 nodes
        let p2 = LayerPartition::auto_partition(32, 2);
        assert_eq!(p2, vec![(0, 16), (16, 32)]);

        // 32 layers across 3 nodes
        let p3 = LayerPartition::auto_partition(32, 3);
        assert_eq!(p3, vec![(0, 11), (11, 22), (22, 32)]);

        // 18 layers (Gemma 2B) across 2 nodes
        let p_gemma = LayerPartition::auto_partition(18, 2);
        assert_eq!(p_gemma, vec![(0, 9), (9, 18)]);
    }

    #[test]
    fn test_layer_partition_parse() {
        let (s, e) = LayerPartition::parse_range("0..16", 32).unwrap();
        assert_eq!((s, e), (0, 16));

        let (s2, e2) = LayerPartition::parse_range("16..32", 32).unwrap();
        assert_eq!((s2, e2), (16, 32));

        let (s3, e3) = LayerPartition::parse_range("0-15", 32).unwrap();
        assert_eq!((s3, e3), (0, 16));
    }

    #[test]
    fn test_swarm_payload_roundtrip() {
        let fwd = ForwardPayload {
            seq_id: 42,
            pos: 10,
            layer_start: 0,
            layer_end: 16,
            dim: 4,
            activations: vec![1.0, 2.5, -3.2, 0.0],
        };
        let encoded = fwd.encode();
        let decoded = ForwardPayload::decode(&encoded).unwrap();
        assert_eq!(decoded.seq_id, 42);
        assert_eq!(decoded.pos, 10);
        assert_eq!(decoded.dim, 4);
        assert_eq!(decoded.activations, vec![1.0, 2.5, -3.2, 0.0]);

        let reply = ReplyPayload {
            seq_id: 100,
            pos: 12,
            token_id: Some(1337),
            activations: vec![0.5, -0.5],
        };
        let rep_enc = reply.encode();
        let rep_dec = ReplyPayload::decode(&rep_enc).unwrap();
        assert_eq!(rep_dec.seq_id, 100);
        assert_eq!(rep_dec.token_id, Some(1337));
        assert_eq!(rep_dec.activations, vec![0.5, -0.5]);
    }

    #[test]
    fn test_loopback_swarm_network() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let server_thread = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let (mtype, payload) = SwarmProto::read_packet(&mut stream).unwrap();
            assert_eq!(mtype, MSG_HANDSHAKE);
            let hs = HandshakePayload::decode(&payload).unwrap();
            assert_eq!(hs.start_layer, 0);

            // Ack handshake
            let ack = HandshakePayload {
                node_id: 1,
                total_nodes: 2,
                start_layer: 16,
                end_layer: 32,
                total_layers: 32,
                dim: hs.dim,
                model_name: hs.model_name,
            };
            SwarmProto::write_packet(&mut stream, MSG_HANDSHAKE_ACK, &ack.encode()).unwrap();

            // Receive forward
            let (mtype, fwd_payload) = SwarmProto::read_packet(&mut stream).unwrap();
            assert_eq!(mtype, MSG_FORWARD);
            let fwd = ForwardPayload::decode(&fwd_payload).unwrap();
            assert_eq!(fwd.pos, 5);

            // Reply
            let rep = ReplyPayload {
                seq_id: fwd.seq_id,
                pos: fwd.pos,
                token_id: None,
                activations: vec![99.0; fwd.dim],
            };
            SwarmProto::write_packet(&mut stream, MSG_REPLY, &rep.encode()).unwrap();
        });

        let mut client = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
        let hs = HandshakePayload {
            node_id: 0,
            total_nodes: 2,
            start_layer: 0,
            end_layer: 16,
            total_layers: 32,
            dim: 4,
            model_name: "test-model".to_string(),
        };
        SwarmProto::write_packet(&mut client, MSG_HANDSHAKE, &hs.encode()).unwrap();

        let (ack_type, ack_buf) = SwarmProto::read_packet(&mut client).unwrap();
        assert_eq!(ack_type, MSG_HANDSHAKE_ACK);
        let ack = HandshakePayload::decode(&ack_buf).unwrap();
        assert_eq!(ack.node_id, 1);

        let fwd = ForwardPayload {
            seq_id: 1,
            pos: 5,
            layer_start: 16,
            layer_end: 32,
            dim: 4,
            activations: vec![1.0, 2.0, 3.0, 4.0],
        };
        SwarmProto::write_packet(&mut client, MSG_FORWARD, &fwd.encode()).unwrap();

        let (rep_type, rep_buf) = SwarmProto::read_packet(&mut client).unwrap();
        assert_eq!(rep_type, MSG_REPLY);
        let rep = ReplyPayload::decode(&rep_buf).unwrap();
        assert_eq!(rep.activations, vec![99.0, 99.0, 99.0, 99.0]);

        server_thread.join().unwrap();
    }
}
