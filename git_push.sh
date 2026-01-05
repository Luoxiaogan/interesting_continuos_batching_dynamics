export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890

git add .
git commit -m "$(cat <<'EOF'
feat(coupling): 目前的theory尝试文档总结和energy分析

目前的难点在于simulation和theory在前期可以完全不对齐，例如simulation可以做大于0的admission但是theory在同一时间点可以做负数的admission。除了第一次simulation eviction，后面diverge之后很难对应来分析。
EOF
)"
git push