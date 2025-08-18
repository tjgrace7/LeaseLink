FROM alpine:3.20
RUN apk add --no-cache curl ca-certificates
WORKDIR /app

# Note: path is relative to the Root Directory (cron), so just "./cron_tick.sh"
COPY ./cron_tick.sh ./cron_tick.sh
RUN sed -i 's/\r$//' ./cron_tick.sh && chmod +x ./cron_tick.sh

CMD ["/bin/sh", "-c", "/app/cron_tick.sh"]
