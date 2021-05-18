from pathlib import Path
import sys
import holoviews as hv

persist_path = Path('..') / 'persist'
project_root = str(Path(sys.path[0]).parents[0].absolute())
project_root
if project_root not in sys.path:
    sys.path.append(project_root)
from scripts import utils

hv.extension('bokeh')

oc_list = [
        '1RAA',
        '1PAC',
        '1PSU',
        '1LRO',
        '1CTR',
        '1BFC',
        '1OUE',
        '1ALO',
        '1NCH',
        '1PNO',
        '1SOU',
#         '1EXP',
#         '1LXF',
#         '1CAP',
        '2BRE',
        '2MPY',
        '2SOU',
        '2RAA',
        '2CTR',    
        '2EST',
        '2NOR',
        '2IDF',
        '2SES',
#         '2CAE',
]

dashboard = utils.webprogress_dashboard(
    orders_path=persist_path /  'orders_all_SV_with_main_origin.pkl',
    clt_path=persist_path / 'clt.pkl',
    filters_include={
        'orgacom': oc_list,
    },
)
dashboard.servable()
