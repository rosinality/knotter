var graph = (function () {
    var s = null;
    var live = false;
    var isLayout = false;
    var selectedNodes = null;

    var functions = {
        clear: function () {
            if (s != null && live) {
                //s.graph.clear();
                s.kill();
                //s.refresh();
                live = false;
            }
        },

        toggleForce: function (toggle, scale, gravity) {
            if (toggle) {
                this.force(scale, gravity);
            } else {
                this.stopForce();
            }
        },

        stopForce: function () {
            if (isLayout) {
                s.killForceAtlas2();
                isLayout = false;
            }
        },

        force: function (scale, gravity) {
            s.startForceAtlas2({worker: true, gravity: gravity,
                scale: scale,
                barnesHutOptimize: false});
            isLayout = true;
        },

        makeGraph: function (container, node, nodeSize, edge, nodeValue, colorScale) {
            var g = {nodes: [], edges: []};
            var nodeValues = {};

            for (var i = 0; i < node; i++) {
                g.nodes.push({
                    id: i,
                    //label: 'Cluster ' + i,
                    x: Math.random(),
                    y: Math.random(),
                    size: nodeSize[i] + 1,
                    color: colorScale(nodeValue[i]),
                });
                nodeValues[i] = nodeValue[i]
            }

            for (var i = 0; i < edge.length; i++) {
                g.edges.push({
                    id: i,
                    source: edge[i][0],
                    target: edge[i][1],
                 //   size: 1,
                    color: colorScale((nodeValues[edge[i][0]] + nodeValues[edge[i][1]]) / 2)
                });
            }

            s = new sigma({
                graph: g,
                container: container,
                settings: {
                    minNodeSize: 2,
                    maxNodeSize: 3
                }
            });

            var selected = null;
            /*sigma.plugins.activateMouseEvents(s,function(err, selectedNodes){
                if (selected != null) {
                    selected.forEach(function (e) {
                        e.color = e.originalColor;
                    });
                }
                selectedNodes.forEach(function (e) {
                    e.originalColor = e.color;
                    e.color = '#76ff03';
                });
                selected = selectedNodes;
                s.refresh();
            });*/
            live = true;
        },

        clearSelection: function () {
            var nodes = s.graph.nodes();

            for (var i = 0; i < nodes.length; i++) {
                if (nodes[i].originalColor) {
                    nodes[i].color = nodes[i].originalColor;
                }
            }
        },

        initialized: function () {
            if (s) {
                return true;
            }

            return false;
        },

        refresh: function () {
            s.refresh();
        },
        
        activateSelection: function (callback) {
            sigma.plugins.activateMouseEvents(s, callback);
        },

        deactivateSelection: function () {
            sigma.plugins.deactivateMouseEvents();
        },

        shuffle: function () {
            var nodes = s.graph.nodes();

            for (var i = 0; i < nodes.length; i++) {
                nodes[i].x = Math.random();
                nodes[i].y = Math.random();
            }

            s.refresh();
        },

        highlightNodes: function (targetNodes) {
            var nodes = s.graph.nodes();

            for (var i = 0; i < targetNodes.length; i++) {
                nodes[targetNodes[i]].color = '#ff0000';
            }

            s.refresh();
        },

        changeColor: function (coloring, colorScale) {
            var nodes = s.graph.nodes();
            var edges = s.graph.edges();
            var newColor = {};

            for (var i = 0; i < nodes.length; i++) {
                nodes[i].color = colorScale(coloring[i]);
                newColor[nodes[i].id] = coloring[i];
            }

            for (var i = 0; i < edges.length; i++) {
                var edge = edges[i];

                edge.color = colorScale((newColor[edge.source] + newColor[edge.target]) / 2);
            }

            s.refresh();
            /*for (var i = 0; i < nodes.length)
            console.log(s.graph.nodes());*/
        },

        makeSampleGraph: function (container) {
            var i,
            s,
            N = 100,
                E = 500,
                g = {
                    nodes: [],
                    edges: []
                };

            // Generate a random graph:
            for (i = 0; i < N; i++) {
                g.nodes.push({
                    id: 'n' + i,
                    label: 'Node ' + i,
                    x: Math.random(),
                    y: Math.random(),
                    size: Math.random(),
                    color: '#666'
                });
            }

            for (i = 0; i < E; i++) {
                g.edges.push({
                    id: 'e' + i,
                    source: 'n' + (Math.random() * N | 0),
                    target: 'n' + (Math.random() * N | 0),
                    size: Math.random(),
                    color: '#ccc'
                });
            }

            // Instantiate sigma:
            s = new sigma({
                graph: g,
                container: container
            });
            var selected = null;
            sigma.plugins.activateMouseEvents(s,function(err, selectedNodes){
                if (selected != null) {
                    selected.forEach(function (e) {
                        e.color = '#666';
                    });
                }
                selectedNodes.forEach(function (e) {
                    e.color = '#ff0000';
                });
                selected = selectedNodes;
                s.refresh();
            });
        }
    };

    return functions;
})();
