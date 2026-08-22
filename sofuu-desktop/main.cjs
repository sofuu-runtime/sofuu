const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1000,
    height: 700,
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#0a0a0a',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  const isDev = process.env.NODE_ENV === 'development';
  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
  } else {
    mainWindow.loadFile(path.join(__dirname, 'dist', 'index.html'));
  }
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// Handle IPC
let activeSofuu = null;

ipcMain.on('chat-request', (event, config) => {
  if (activeSofuu) {
    activeSofuu.kill();
  }

  // Create the sofuu process
  const isDev = process.env.NODE_ENV === 'development';
  const sofuuPath = isDev ? path.resolve(__dirname, '..', 'sofuu') : path.join(process.resourcesPath, 'sofuu');
  const bridgePath = isDev ? path.resolve(__dirname, 'sofuu-bridge.js') : path.join(process.resourcesPath, 'sofuu-bridge.js');
  
  const env = { ...process.env };
  if (config.apiKey) {
    // Determine the likely environment variable name for the provider
    const p = config.provider.toUpperCase();
    env[`${p}_API_KEY`] = config.apiKey;
  }

  const os = require('os');
  const fs = require('fs');
  config.homedir = os.homedir();
  
  const brainDir = path.join(config.homedir, '.sofuu_brains');
  if (!fs.existsSync(brainDir)) fs.mkdirSync(brainDir, { recursive: true });

  activeSofuu = spawn(sofuuPath, ['run', bridgePath], { env });

  // Send the config as JSON string to the bridge script via stdin
  activeSofuu.stdin.write(JSON.stringify(config) + '\n');
  activeSofuu.stdin.end();

  let leftover = '';
  let errorOccurred = false;

  activeSofuu.stdout.on('data', (data) => {
    leftover += data.toString();
    const lines = leftover.split('\n');
    leftover = lines.pop(); // keep the last line if it's incomplete
    
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const msg = JSON.parse(line);
        if (msg.type === 'chunk') {
          event.reply('chat-chunk', msg.content);
        } else if (msg.type === 'done') {
          event.reply('chat-done', msg.stats);
        } else if (msg.type === 'error') {
          errorOccurred = true;
          event.reply('chat-error', msg.message);
        }
      } catch (err) {
        console.error('Failed to parse stdout line from Sofuu:', line);
      }
    }
  });

  activeSofuu.stderr.on('data', (data) => {
    console.error(`[Sofuu stderr]: ${data}`);
  });

  activeSofuu.on('close', (code) => {
    if (leftover.trim()) {
      try {
        const msg = JSON.parse(leftover);
        if (msg.type === 'chunk') {
          event.reply('chat-chunk', msg.content);
        } else if (msg.type === 'error') {
          errorOccurred = true;
          event.reply('chat-error', msg.message);
        }
      } catch (e) {}
    }
    
    if (!errorOccurred) {
      event.reply('chat-done', {}); // fallback if done wasn't sent
    }
    activeSofuu = null;
  });
});
