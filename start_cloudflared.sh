#!/bin/bash
truncate -s 0 /root/cloudflared.log
cloudflared tunnel --url http://localhost:7860 >> /root/cloudflared.log 2>&1
