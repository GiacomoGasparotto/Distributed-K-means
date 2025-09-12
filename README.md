# Distributed-K-means

## CloudVeneto

1. Create `.ssh/config` file:
```
### CLOUD VENETO

# gate

Host cloudveneto
    HostName gate.cloudveneto.it
    User edamore
    IdentityFile ~/private/personal/key/location

# MAPD-B nodes

Host master
    HostName 10.67.22.224
    User ubuntu
    IdentityFile ~/private/project/key/location
    ProxyJump cloudveneto
    LocalForward 8080 localhost:8080
    LocalForward 4040 localhost:4040

Host worker1
    HostName 10.67.22.170
    User ubuntu
    IdentityFile ~/private/personal/key/location
    ProxyJump cloudveneto

Host worker2
    HostName 10.67.22.202
    User ubuntu
    IdentityFile ~/private/project/key/location
    ProxyJump cloudveneto

Host worker3
    HostName 10.67.22.208
    User ubuntu
    IdentityFile ~/private/personal/key/location
    ProxyJump cloudveneto
```
<span style="color: red; font-weight: bold">WARNING:</span> remember to change the `User` and `IdentityFile` fields for the `cloudveneto` host with your personal information.

2. Connect to master:
```bash
ssh master
```
You will be prompted to provide your CloudVeneto password in order to connect.

3. (Optional) Open tmux session called `mapd`:
```bash
tmux new -As mapd # the flag -As tells tmux to attach to an existing `mapd` session or create a new one if it doesn't exist
```
Useful tmux commands:
```bash
tmux new -s my_sess # creates a new session called `my_sess`
tmux ls # lists all active sessions
tmux a # attach to last opened session
tmux a -t my_sess # opens the session `my_sess`
tmux kill-session -t my_sess # deletes the session `my_sess`
tmux kill-server # deletes all sessions and kills the tmux server
```
Once inside a session (`<C-b>` stands for `Ctrl + b` or command idk how it works on Mac):
```tmux
<C-b> c # creates a new window
<C-b> & # deletes the current window (after confirmation)
<C-b> n # moves to next window
<C-b> p # moves to next window
<C-b> w # see all windows
<C-b> s # see all sessions
<C-b> d # detaches from the session (it is still active)
<C-b> : # enter command mode (basically can run commands as seen before)
```
tmux is useful as the sessions survive an eventual disconnection and can be re-attached to.

4. Activate environment if not already active:
```bash
source .venv/bin/activate
```

5. Start cluster:
```bash
$SPARK_HOME/sbin/start-all.sh
```

6. Open `VSCode` on local machine and run the command `Remote-SSH: Connect to Host` selecting `master`.
You will be prompted to provide your CloudVeneto password in order to connect.
Open the desired folder.

7. Open your browser and go to `localhost:8080` and `localhost:4040` in order to monitor the spark session.

8. Once all operations are finished, stop the cluster:
```bash
$SPARK_HOME/sbin/stop-all.sh
```

## Docker

+ Compose with build:: Docker builds images from your local Dockerfiles (or private repo e.g. ´jpazzini/mapd-b:spark-worker´).

Build And Run

+ Build locally: docker compose build --no-cache

+ Start services: docker compose up -d --scale spark-worker=3

+ Jupyter NB UI: http://localhost:1234

+ Spark UIs: Master http://localhost:8080, Driver http://localhost:4040