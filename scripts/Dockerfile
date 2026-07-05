FROM pgvector/pgvector:pg16

# Bake the init script into the image so we don't depend on Docker's
# file-mount behavior (which can create a directory instead of a file
# if the source file doesn't exist on the host at first run).
COPY init_db.sql /docker-entrypoint-initdb.d/00_init.sql
