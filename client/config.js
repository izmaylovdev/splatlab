// SplatLab client configuration.
//
// Empty base = talk to the same origin that served this page (the co-located /
// API-served deployment). To host this client separately from the API, set both
// to the API's public origin, e.g.:
//
//   window.SPLATLAB_API_BASE = 'https://splat-api.example.com';
//   window.SPLATLAB_WS_BASE  = 'wss://splat-api.example.com';
//
// If only API_BASE is set, the WebSocket base is derived from it automatically.
window.SPLATLAB_API_BASE = window.SPLATLAB_API_BASE || '';
window.SPLATLAB_WS_BASE = window.SPLATLAB_WS_BASE || '';
