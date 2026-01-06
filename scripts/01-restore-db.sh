#!/bin/sh
set -e

# Restore the database from the init-db.dump file
if [ -f /docker-entrypoint-initdb.d/init-db.dump ]; then
    pg_restore --no-owner --no-privileges -U "$POSTGRES_USER" -d "$POSTGRES_DB" /docker-entrypoint-initdb.d/init-db.dump
else
    echo "init-db.dump file not found, skipping database restore"
fi
