const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  sendChat: (config) => ipcRenderer.send('chat-request', config),
  onChatChunk: (callback) => ipcRenderer.on('chat-chunk', (_event, value) => callback(value)),
  onChatDone: (callback) => ipcRenderer.on('chat-done', (_event, stats) => callback(stats)),
  onChatError: (callback) => ipcRenderer.on('chat-error', (_event, error) => callback(error)),
  removeAllListeners: () => {
    ipcRenderer.removeAllListeners('chat-chunk');
    ipcRenderer.removeAllListeners('chat-done');
    ipcRenderer.removeAllListeners('chat-error');
  }
});
