package web

import "embed"

// Assets embeds all web frontend assets for the Nexus Cortex dashboard.
//go:embed index.html index.css index.js
var Assets embed.FS
