#!/bin/sh

rclone copy -P --exclude ".venv/" --exclude ".git/" --exclude "__pycache__/" --exclude "wandb/" --exclude "logs/" ./ sp-collab:/post-sparse
