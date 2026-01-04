export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890

git add .
git commit -m" 添加了`simulation_admission_control_random`, 作为single type admission测试小s的threshold实验中, 使用gamma分布来实现stochastic的问题. 具体要查看`simulation_admission_control_random/实现细节.md`和`simulation_admission_control_random/使用说明.md`."
git push