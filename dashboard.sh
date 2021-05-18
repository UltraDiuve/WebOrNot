conda activate py38
cd /homedm/pmasse/WebOrNot/scripts/
panel serve /homedm/pmasse/WebOrNot/scripts/dashboard_script.py --port=8080 --allow-websocket-origin=devdm:8080
