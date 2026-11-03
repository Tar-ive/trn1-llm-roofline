#!/bin/bash
# cloud-init user-data, rendered by infra/launch.sh. Runs once as root at first boot.
# Log: /var/log/cloud-init-output.log
MINUTES=__MINUTES__

# Hard stop first, so nothing below can leave the box running past budget.
# With shutdown-behavior=terminate, a poweroff terminates the instance (and its EBS).
shutdown -P +$((MINUTES + 2)) "neuron-lab budget reached"

set -euo pipefail
echo $(( $(date +%s) + MINUTES * 60 )) > /etc/neuron-lab-deadline
printf 'NEURON_LAB_MODE=%s\nNEURON_LAB_BUCKET=%s\n' "__MODE__" "__BUCKET__" > /etc/neuron-lab.env

U=ubuntu; H=/home/$U; W=$H/work

# --- git access via the repo-scoped deploy key (written before `set -x` so it never hits the console log) ---
install -d -m 700 -o $U -g $U $H/.ssh
echo '__DEPLOY_KEY_B64__' | base64 -d > $H/.ssh/deploy_key
set -x
cat > $H/.ssh/config <<'EOF'
Host github.com
  IdentityFile ~/.ssh/deploy_key
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
EOF
chmod 600 $H/.ssh/deploy_key $H/.ssh/config
chown $U:$U $H/.ssh/deploy_key $H/.ssh/config

runuser -u $U -- git clone -b __BRANCH__ git@github.com:__REPO__.git $W
runuser -u $U -- git -C $W config user.name "__GIT_NAME__ (trn)"
runuser -u $U -- git -C $W config user.email "__GIT_EMAIL__"

# --- autosave: commit everything (minus >50MB files) and push, with retries ---
cat > /usr/local/bin/autosave <<'EOF'
#!/bin/bash
cd /home/ubuntu/work || exit 0
find . -path ./.git -prune -o -type f -size +50M -print | sed 's|^\./||' | while read -r f; do
  grep -qxF "$f" .git/info/exclude || echo "$f" >> .git/info/exclude
done
git add -A
git commit -qm "${1:-autosave}: $(hostname) $(date -u +%FT%TZ)" || true
for i in 1 2 3; do
  git pull -q --rebase -X theirs && git push -q && exit 0
  sleep 5
done
echo "autosave: push failed" >&2; exit 1
EOF
chmod 755 /usr/local/bin/autosave

# every 10 minutes
echo "*/10 * * * * $U /usr/local/bin/autosave periodic >> $H/autosave.log 2>&1" > /etc/cron.d/autosave

# on any shutdown/reboot (ExecStop runs before the network goes down)
cat > /etc/systemd/system/autosave-on-shutdown.service <<'EOF'
[Unit]
Description=Push work to git before shutdown
Wants=network-online.target
After=network-online.target
[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/true
ExecStop=/usr/sbin/runuser -u ubuntu -- /usr/local/bin/autosave shutdown
TimeoutStopSec=120
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now autosave-on-shutdown.service

# final save at T-5min, then power off (=> terminate). `shutdown +N+2` above is the backstop.
systemd-run --unit=neuron-lab-final --on-active=$((MINUTES - 5))min \
  /bin/bash -c "runuser -u $U -- /usr/local/bin/autosave final; systemctl poweroff"

runuser -u $U -- /usr/local/bin/autosave "boot"

if [ "__AUTORUN__" = "1" ]; then
  runuser -u $U -- bash -lc "cd $W && nohup bench/run.sh > $H/bench-nohup.log 2>&1 &"
fi
