# This file is part of fast-wikification.
#
# Fast-wikification is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.

# Fast-wikification is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.

# You should have received a copy of the GNU General Public License along with
# fast-wikification. If not, see <https://www.gnu.org/licenses/>.

import json
import os

CDIR = os.path.dirname(os.path.abspath(__file__))


def load_jsonl(fpath):
    """
    Loads and returns jsonl in list format.
    :param fpath:
    :return:
    """
    with open(fpath, 'r') as f:
        return [json.loads(ln) for ln in f.readlines()]


def restructure_dataset(dataset):
    """
    Re-formats the dataset to the following format:
    {
        "context_left":"...",
        "mention":"...",
        "context_right":"...",
        "query_id":"...",
        "label_id":...,
        "Wikipedia_ID":...,
        "Wikipedia_URL":"...",
        "Wikipedia_title":"...",
        "mention_position":...,
        "mention_length":...,
    }
    :param dataset:
    :return:
    """
    restructured_dataset = []
    for d in dataset:
        d['Wikipedia_ID'] = int(d['Wikipedia_ID']) if d['Wikipedia_ID'] else -1
        d['mention_position'] = len(d['context_left'].split())
        d['mention_length'] = len(d['mention'].split())
        restructured_dataset.append(d)
    return restructured_dataset


testa = restructure_dataset(
    load_jsonl(os.path.join(CDIR, 'blink', 'data', 'BLINK_benchmark', 'AIDA-YAGO2_testa.jsonl'))
)
testb = restructure_dataset(
    load_jsonl(os.path.join(CDIR, 'blink', 'data', 'BLINK_benchmark', 'AIDA-YAGO2_testb.jsonl'))
)
wnedwiki = restructure_dataset(
    load_jsonl(os.path.join(CDIR, 'blink', 'data', 'BLINK_benchmark', 'wnedwiki_questions.jsonl'))
)
clueweb = restructure_dataset(
    load_jsonl(os.path.join(CDIR, 'blink', 'data', 'BLINK_benchmark', 'clueweb_questions.jsonl'))
)

datasets = {
    'AIDA-YAGO2-testa': testa,
    'AIDA-YAGO2-testb': testb,
    'Clueweb': clueweb,
    'WNEDWiki': wnedwiki
}
print('loaded...')

