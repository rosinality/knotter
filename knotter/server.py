import asyncio
import threading
import traceback
import aiohttp
from aiohttp import web
import json
import pandas as pd
from io import StringIO
import numpy as np
import scipy as sp
import scipy.linalg as la
from scipy import stats
import itertools
import functools
import jinja2
import collections
import os

from knotter.tda import *
import knotter.knitter

class Analyzer:
    def __init__(self):
        self.df = None
        self.projection = None
        self.lense_map = {'Principal Component': pca_projection,
                'Linfty Centering': Linfty_centering,
                'Gaussian Density': gaussian_density,
                'Nearest Neighbor': nearest_neighbor,
                't-SNE': t_SNE}

    def load_dataframe(self, data_string):
        data_io = StringIO(data_string)
        self.df = pd.read_csv(data_io)
                
    def variable_list(self):
        return self.df.columns.values

    def variable_summary(self):
        df_min = self.df_m.min(axis=0)
        df_max = self.df_m.max(axis=0)
        df_mean = self.df_m.mean(axis=0)
        df_std = self.df_m.std(axis=0)

        descriptive = self.df.describe()

        lense_min = self.projection.min(axis=0)
        lense_max = self.projection.max(axis=0)
        lense_mean = self.projection.mean(axis=0)
        lense_std = self.projection.std(axis=0)

        result = []

        for no, (m, M, u, s) in enumerate(zip(lense_min, lense_max,
            lense_mean, lense_std)):
            result.append(['Lense {}'.format(no + 1), float(m), float(M), float(u), float(s)])

        #for no, m, M, u, s in zip(self.variable_list(), df_min,
        #        df_max, df_mean, df_std):
        for no, m, M, u, s in zip(descriptive.loc['count'].index, descriptive.loc['min'],
                descriptive.loc['max'], descriptive.loc['mean'], descriptive.loc['std']):
            result.append([no, float(m), float(M), float(u), float(s)])

        return result


    def lense_change(self, variables, lenses):
        for dtype in self.df[variables].dtypes:
            if dtype.kind not in 'biufc':
                raise ValueError('Variable list contains non-numeric column')

        self.df_m = self.df[variables].dropna().as_matrix()
        self.lense_type = []
        explained_variance = None

        oneshot_lense = ['Principal Component', 't-SNE', 'Nearest Neighbor']
        oneshot_num = dict(zip(oneshot_lense, [0] * len(oneshot_lense)))

        pca_num = 0
        t_sne_num = 0
        sp_var = []
        for lense in lenses:
            lense_type = lense['lenseProperty']['type']

            if lense_type in oneshot_lense:
                oneshot_num[lense_type] += 1

            elif lense_type == 'Simple Projection':
                if 'variable' in lense['lenseProperty'] and \
                        lense['lenseProperty']['variable'].strip():
                    sp_var.append(lense['lenseProperty']['variable'])

            self.lense_type.append(lense_type)

        oneshot_projection = {}
        for k, v in oneshot_num.items():
            if v < 1:
                continue

            if k == 'Principal Component':
                proj, explained_variance = self.lense_map[k](
                        self.df_m, n_components=v)

            else:
                proj = self.lense_map[k](self.df_m, n_components=v)

            oneshot_projection[k] = proj

        if len(sp_var) > 0:
            variables2 = variables.copy()
            
            for var in sp_var:
                if var not in variables2:
                    variables2.append(var)

            df2 = self.df[variables2].dropna()
            self.df_m = df2[variables].as_matrix()
            df2 = df2[sp_var]

        projections = []

        oneshot_nth = dict(zip(oneshot_lense, [0] * len(oneshot_lense)))
        import traceback
        try:
            for lense in lenses:
                lense_type = lense['lenseProperty']['type']

                if lense_type in oneshot_lense:
                    projections.append(oneshot_projection[lense_type] \
                            [:, oneshot_nth[lense_type]])
                    oneshot_nth[lense_type] += 1

                elif lense_type == 'Simple Projection':
                    if not 'variable' in lense['lenseProperty']:
                        projections.append(np.zeros(self.df_m.shape[0]))
                        continue
                    var_name = lense['lenseProperty']['variable']
                    if not var_name.strip(): 
                        projections.append(np.zeros(self.df_m.shape[0]))
                        continue
                        
                    projections.append(df2[var_name].as_matrix())

                else:
                    projections.append(self.lense_map[lense_type] \
                            (self.df_m, lense['lenseProperty']))

        except:
            print('Wrong Lense Type')
            traceback.print_exc()

            return

        #print(projections[0].shape)
        self.projection = np.vstack(projections).T
        if not isinstance(explained_variance, type(None)):
            self.explained_variance = explained_variance

        self.covers = []
        for no, lense in enumerate(lenses):
            self.covers.append(self._cover_change(no, int(lense['cover']['no']),
                float(lense['cover']['overlap']), lense['cover']['balanced']))
            
    def cover_change(self, covers):
        if isinstance(self.projection, type(None)):
            return

        #print('# of covers: ' + str(len(covers)))

        self.covers = []
        for no, cover in enumerate(covers):
            self.covers.append(self._cover_change(no, int(cover['no']),
                float(cover['overlap']), cover['balanced']))

    def _cover_change(self, no, cover_no, overlap, balanced):
        if balanced:
            return balanced_cover(self.projection[:, no], cover_no, overlap)

        else:
            return uniform_cover(self.projection[:, no].min(),
                    self.projection[:, no].max(),
                    cover_no, overlap)

    def lense_summary(self):        
        result = []
        #print(self.projection.shape)
        _, dim = self.projection.shape
        pca_nth = 0

        for i in range(dim):
            data = {}
            data['min'] = self.projection[:, i].min()
            data['max'] = self.projection[:, i].max()
            data['size'] = abs(self.covers[i][0, 1] - self.covers[i][0, 0])

            if self.lense_type[i] == 'Principal Component':
                data['explained_variance'] = self.explained_variance[pca_nth]
                pca_nth += 1

            result.append(data)

        return result

    def variable_from_cluster(self, no):
        result = []

        for c in self.cluster:
            values = []
        
            for i in c:
                values.append(self.df[no].iloc[i])
                
            result.append(values)

        return result

    def variable_coloring(self, no):
        values = self.variable_from_cluster(no)

        return mean_lense(values)

    def lense_coloring(self, no):
        lense_values = lense_from_cluster(self.cluster, self.projection, nth=no)

        return mean_lense(lense_values)

    def count_points(self, nodes):
        return len(self.points_in_nodes(nodes))
    
    def points_in_nodes(self, nodes):
        points = set()

        for n in nodes:
            points = points.union(self.cluster[n])

        return points

    def show_node(self, groups, label):
        point_group = collections.OrderedDict()

        for g in groups:
            no = g['no']
            nodes = g['nodes']
            points = set()

            for n in nodes:
                points = points.union(self.cluster[n])

            point_group[no] = self.df[label].iloc[list(points)]

        return point_group.keys(), itertools.zip_longest(*point_group.values(), fillvalue='')

    def geography(self, groups, latitude, longitude, label):
        #color = ['red', 'green', 'blue', 'yellow', 'black']

        point_group = {}
        i = 0
        for g in groups:
            no = g['no']
            nodes = g['nodes']
            points = set()

            for n in nodes:
                points = points.union(self.cluster[n])

            for n in points:
                point_group[i] = [self.df[latitude].iloc[n], self.df[longitude].iloc[n], no - 1, self.df[label].iloc[n]]
                i += 1

        return point_group

    def find_point(self, label, point):
        key = self.df[self.df[label] == point]
        if key.size > 0:
            found = key.index[0]
        result = []
        for n, c in enumerate(self.cluster):
            if found in c:
                result.append(n)

        return result

    def descriptive_compare(self, groups):
        point_group = collections.OrderedDict()

        for g in groups:
            no = g['no']
            nodes = g['nodes']
            points = set()

            for n in nodes:
                points = points.union(self.cluster[n])

            point_group[no] = self.df.iloc[list(points)].describe()

        table = []
        for agg in ('min', 'mean', '50%', 'max'):
            sub_col = []
            for v in point_group.values():
                sub_col.append(v.T[agg].values)
            table.append(np.vstack(sub_col).T)
        a_table = np.hstack(table)

        headers = []
        for i, j in itertools.product(('min', 'mean', 'median', 'max'),
                point_group.keys()):
            headers.append('{} {}'.format(j, i))

        return headers, zip(point_group[no].T.index, a_table)

    def analyze(self, bins, metric):
        points = point_in_intervals(self.projection, self.covers)
        cluster = make_cluster(self.df_m, points, metric=metric)
        self.cluster = cut_cluster(cluster, cut_points(points),
                        threshold=cutoff_histogram(bins=bins, nth=-1))
        links = find_nerves(self.cluster)

        return self.cluster, links, self.projection[:, 0].tolist(), self.lense_coloring(0)

    def analyze2(self, bins):
        G = knitter.eps_graph(self.df_m, bins)
        _, self.cluster, links = knitter.process(G)
        #import networkx as nx
        #g = nx.Graph(G)
        #self.cluster = g.nodes()
        #links = g.edges()
        link = []
        for l in links:
            link.append((int(l[0]), int(l[1])))
        
        return self.cluster, link, self.projection[:, 0].tolist(), self.projection[:, 0].tolist()

class Server:
    def __init__(self):
        self.analyzer = Analyzer()

        here = os.path.abspath(os.path.dirname(__file__))
        self.env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(os.path.join(here, 'templates')))

        self.descriptive_report = self.env.get_template('descriptive.html')
        self.node_report = self.env.get_template('node-report.html')

        self.event_map = {
                'connect': self.connect,
                'upload_data': self.read_data,
                'variable_histogram': self.variable_histogram,
                'lense_change': self.lense_change,
                'cover_change': self.cover_change,
                'analyze': self.analyze,
                'coloring': self.coloring,
                'node_info': self.node_info,
                'compare_node': self.compare_node,
                'show_node': self.show_node,
                'geography': self.geography,
                'find_point': self.find_point
                }

    def analyze2(self, data):
        cluster, links, lense, mean_lense = self.analyzer.analyze2(float(data['bins']))
        #node_size = cluster_size(cluster)
        #node_size /= node_size.max()
        node_size = np.ones(len(cluster))
        #print(type(links[0][0]))
        print(len(cluster))
        self.response('main_analysis_result',
                {'node': len(cluster),
                    'link': links,
                    'lense': lense,
                    'nodeSize': node_size.tolist(),
                    'meanLense': mean_lense,
                    'summary': self.analyzer.variable_summary()})

    def response(self, key, content):
        content['type'] = key
        self.ws.send_str(json.dumps(content))

    def connect(self, data):
        pass

    def find_point(self, data):
        self.response('find_point', {'nodes':
            self.analyzer.find_point(data['label'], data['point'])})

    def read_data(self, data):
        self.analyzer.load_dataframe(data['content'])
        self.response('variable_list',
                {'content': list(self.analyzer.variable_list())})

    def node_info(self, data):
        self.response('node_info',
                {'group': data['group'],
                    'count': self.analyzer.count_points(data['nodes'])})

    def geography(self, data):
        result = self.analyzer.geography(data['group'], data['latitude'], data['longitude'], data['label'])
        self.response('geography', {'coord': result})

    def show_node(self, data):
        groups, rows = self.analyzer.show_node(data['group'],
                data['label'])
        self.response('analysis_report',
                {'content': self.node_report.render(
                    groups=groups, rows=rows)})

    def compare_node(self, data):
        headers, rows = self.analyzer.descriptive_compare(data['group'])
        self.response('analysis_report',
                {'content':
                    self.descriptive_report.render(
                        headers=headers, rows=rows)})

    def variable_histogram(self, data):
        self.response('variable_histogram',
                {'content': list(self.analyzer.df[data['content']].dropna()
                    .as_matrix().tolist())})

    def lense_change(self, data):
        try:
            self.analyzer.lense_change(data['variables'], data['lenses'])
            self.response('lense_summary',
                {'content': self.analyzer.lense_summary()})

        except:
            pass
        
    def cover_change(self, data):
        self.analyzer.cover_change(data['covers'])
        self.response('lense_summary',
                {'content': self.analyzer.lense_summary()})

    def coloring(self, data):
        num_lenses = self.analyzer.projection.shape[1]
        no = data['no']

        if no.startswith('Lense '):
            index = int(no.replace('Lense ', '')) - 1
            values = self.analyzer.projection[:, index]
            self.response('coloring',
                    {'coloring': self.analyzer.lense_coloring(index),
                        'values': values.tolist(),
                        'min': values.min().tolist(),
                        'max': values.max().tolist()})

        else:
            #no -= num_lenses
            values = np.nan_to_num(self.analyzer.df[no].values)
            self.response('coloring',
                    {'coloring': self.analyzer.variable_coloring(no),
                        'values': values.tolist(),
                        'min': values.min().tolist(),
                        'max': values.max().tolist()})

    def analyze(self, data):
        metric_label = {'Euclidean L2': 'euclidean',
                'Minkowski L1': 'minkowski',
                'Chebyshev L∞': 'chebyshev',
                'Cosine': 'cosine',
                'Correlation': 'correlation',
                'Hamming': 'hamming',
                'Standardized Euclidean': 'seuclidean'}
        cluster, links, lense, mean_lense = self.analyzer.analyze(
                int(data['bins']), metric=metric_label[data['metric']])
        node_size = cluster_size(cluster)
        node_size /= node_size.max()

        self.response('main_analysis_result',
                {'node': len(cluster),
                    'link': links,
                    'lense': lense,
                    'nodeSize': node_size.tolist(),
                    'meanLense': mean_lense,
                    'summary': self.analyzer.variable_summary()})

    async def handler(self, request):
        self.ws = web.WebSocketResponse()
        await self.ws.prepare(request)

        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)

                try:
                    self.event_map[data['type']](data)

                except KeyError:
                    print('Wrong message type: {}'.format(data['type']))
                    traceback.print_exc()

                except:
                    traceback.print_exc()
                    
                #print('Got message: {}'.format(data['type']))

            elif msg.type == aiohttp.WSMsgType.BINARY:
                print('Got Binary')
            
            elif msg.type == aiohttp.WSMsgType.CLOSE:
                await ws.close()

        return self.ws

@asyncio.coroutine
def ws_handler(request):
    ws = web.WebSocketResponse()
    ws.start(request)

    while True:
        msg = yield from ws.receive()

        if msg.tp == web.MsgType.text:
            data = json.loads(msg.data)

            if data['type'] == 'upload_data':
                data_io = StringIO(data['content'])
                df = pd.read_csv(data_io)
                response = {'type': 'variable_list',
                        'content': list(df.columns.values)}
                ws.send_str(json.dumps(response))

            print('Got message: {}'.format(data['type']))

        elif msg.tp == web.MsgType.binary:
            print('Got Binary')
            
        elif msg.tp == web.MsgType.close:
            break

    return ws

@asyncio.coroutine
def init(loop):
    appServer = Server()
    app = web.Application(loop = loop)
    app.router.add_route('GET', '/ws', appServer.handler)

    server = yield from loop.create_server(app.make_handler(),
            '127.0.0.1', 9000)
    print('Server started at http://localhost:9000')

    return server

def stop_loop():
    input('Press <enter> to stop\n')
    loop.call_soon_threadsafe(loop.stop)

def run():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(init(loop))
    #threading.Thread(target = stop_loop).start()
    loop.run_forever()
    loop.close()

if __name__ == '__main__':
    run()
