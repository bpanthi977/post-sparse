#!/bin/sh

rclone copy -P --exclude ".venv/" --exclude ".git/" --exclude "__pycache__/" ./ sp-collab:/post-sparse
