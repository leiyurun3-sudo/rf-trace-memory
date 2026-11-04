R74_SplitDebt — 自动探测 R73 目录，生成 48 个 job（6 任务 x 2 seed x 4 臂）

臂: DISC_ONLY / COND_ONLY / GATE_FADE / DISC_FADE
原理: c = (p + f_empty) + (f_src - f_empty) = c_disc + c_cond
      反演已算过 f_empty，零额外前向

用法（在 MINT_Flow_Jupyter_Ready 根目录）:
  unzip -oq /root/autodl-tmp/R74_SplitDebt_Update.zip -d .
  python3 install_r74.py
  bash <打印出来的路径>/run_all.sh
