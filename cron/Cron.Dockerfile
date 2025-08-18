FROM alpine:3.20

# Install curl (tiny image, quick boot)
RUN apk add --no-cache curl

WORKDIR /app
COPY scripts/cron_tick.sh /app/cron_tick.sh
RUN chmod +x /app/cron_tick.sh

# Render runs the container on the schedule and executes this CMD
CMD ["/app/cron_tick.sh"]
