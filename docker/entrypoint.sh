#!/bin/bash
set -e

# Ensure directories exist with proper permissions for all users
mkdir -p /app/outputs /app/models /app/data
chmod -R 777 /app/outputs /app/models /app/data

# Execute the command
exec "$@"
