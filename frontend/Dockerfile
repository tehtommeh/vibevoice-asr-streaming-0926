# Frontend: static single-page demo served by nginx, which also proxies the API.
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY web/ /usr/share/nginx/html/

EXPOSE 80

# 127.0.0.1, not localhost: nginx binds IPv4 only, and BusyBox wget tries the
# ::1 that /etc/hosts lists for localhost first -- which refuses the connection.
HEALTHCHECK --interval=15s --timeout=4s --start-period=5s --retries=5 \
  CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
