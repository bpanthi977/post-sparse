#!/bin/sh

rclone copy -P --exclude ".venv/" --exclude ".git/" --exclude "data/coco/val2017/" --exclude "__pycache__/" sp-collab:/post-sparse ./
