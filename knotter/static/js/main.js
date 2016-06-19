(function (document) {
    'use strict';

    var $prototype = function (element, selector) {
        var key = selector.charAt(0);
        var rest = selector.substr(1);

        switch (key) {
            case '#':
                return element.getElementById(rest);

            case '.':
                return element.getElementsByClassName(rest);

            default:
                return element.getElementsByTagName(rest);
        }
    };

    var showOption = function (selector, index) {
        if (selector == index) {
            return false;
        }

        return true;
    };

    var range = function (end) {
        return Array.apply(null, Array(end)).map(function (_, i) {return i;});
    };

    var $ = function (selector) {
        return $prototype(document, selector);
    };

    var initApp = function (app) {
        app.showOption = showOption;
        app.exponent = '1.0';
        app.lense = 0;
        app.lenseMin = 0;
        app.lenseMax = 0;
        app.variables = [];
        app.selectedVariables = [];
        app.clusteringHistogramBins = 30;
        app.layoutActive = false;
        app.postAnalysisPanel = false;
        app.coloringVariable = 0;
        app.coloringSummary = [];
        app.comparingGroup = [{name: 1}, {name: 2}, {name: 3}];
        app.checkedGroup = [];
        app.nodeLabeling = '';
        app.forceScale = 1;
        app.forceGravity = 1;
        app.metric = 'Euclidean L2';
    };

    var receive = function (msg) {
        var data = JSON.parse(msg.data);

        var messages = {
            variable_list: function (data) {
                app.variables = data.content;
            },

            variable_histogram: function (data) {
                d3.selectAll('#histogram > *').remove();
                d3plot.makeHistogram('#histogram', data.content, 100);
            },

            lense_range: function (data) {
                app.lenseMin = data.content[0];
                app.lenseMax = data.content[1];
            },

            main_analysis_result: function (data) {
                d3.selectAll('#histogram > *').remove();
                d3plot.makeHistogram('#histogram', data.lense, 100);
                graph.clear();
                graph.makeGraph('main-graph', data.node, data.nodeSize, data.link,
                        data.meanLense, d3plot.scale);
                app.coloringSummary = data.summary;
            },

            coloring: function (data) {
                d3.selectAll('#histogram > *').remove();
                d3plot.makeHistogram('#histogram', data.values, 100);
                graph.changeColor(data.coloring, d3plot.scale);
            },

            lense_summary: function (data) {
                for (var i = 0; i < app.lenses.length; i++) {
                    app.set('lenses.' + i + '.lenseMin', data.content[i].min);
                    app.set('lenses.' + i + '.lenseMax', data.content[i].max);
                    app.set('lenses.' + i + '.intervalSize', data.content[i].size);

                    if (data.content[i].explained_variance) {
                        app.set('lenses.' + i + '.explainedVariance',
                                data.content[i].explained_variance * 100);
                    }
                }
            },

            node_info: function (data) {
                app.set('groups.' + data.group + '.points', data.count);
            },

            analysis_report: function (data) {
                $('#analysis-report').innerHTML = data.content;
            },

            geography: function (data) {
                geography.createMap($('#analysis-report'), 37.42574929824175, 126.76700465024426);
                geography.ready(function () { geography.makeMarker(data.coord); });
            },

            find_point: function (data) {
                graph.highlightNodes(data.nodes);
            }
        };

        messages[data.type](data);
    };

    var initWebSocket = function (ws) {
        ws.onerror = function (error) {
            $('#toast-connection-error').show();
        };
        ws.onclose = function () { };
        ws.onopen = function () {
            var msg = JSON.stringify({'type': 'connect', 'content': 'HELLO SERVER'});
            ws.send(msg);
            //$('#status').innerHTML = '<span style="color: #7cb342;">CONNECTED</span>';
        };
        ws.onmessage = receive;
    };

    var ws = new WebSocket('ws://localhost:9000/ws');
    initWebSocket(ws);

    var sendEvent = function (type, data) {
        data.type = type;
        ws.send(JSON.stringify(data))
    }

    Element.prototype.$ = function (selector) { return $prototype(this, selector); };
    Element.prototype.on = function (name, event_function) {
        this.addEventListener(name, event_function);
    };

    var app = $('#app');
    initApp(app);

    app.fixed = function (value) {
        return value.toFixed(3);
    };


    var selectedVariable = function () {
        var selected = [];
        for (var i = 0; i < app.selectedVariables.length; i++) {
            selected.push(app.variables[app.selectedVariables[i]]);
        }
        return selected;
    }

    app.check = function (e) {
        console.log('Event Fired (Check)');
    };

    app.lenseChange = function (e) {
        var selected = selectedVariable();

        if (selected.length > 0) {
            sendEvent('lense_change', {variables: selected,
                lenses: e.detail.lenses});
        }
    };

    app.variableSelected = function (e) {
        if (!app.selectedVariables) {
            return;
        }
        //console.log('Fired');
        var selected = selectedVariable();
        if (selected.length > 0) {
            sendEvent('lense_change', {variables: selected, lenses: app.lenses});
        }
    };

    app.coverChange = function (e) {
        var selected = selectedVariable();

        if (selected.length < 1) {
            return;
        }

        var covers = [];

        for (var i = 0; i < e.detail.lenses.length; i++) {
            covers.push({no: e.detail.lenses[i].cover.no,
                overlap: e.detail.lenses[i].cover.overlap,
                balanced: e.detail.lenses[i].cover.balanced});
        }
        sendEvent('cover_change', {covers: covers});
    };

    var showToast = function (msg) {
        app.toastMessage = msg;
        $('#toast-general').show();
    };

    var mainGraphLayout = false;

    app.graphLayout = function () {
        graph.toggleForce(app.layoutActive, app.forceScale, app.forceGravity);
    };

    app.graphShuffle = function () {
        graph.shuffle();
    };

    var selectedNodes = {};

    var groupColorNode = function (index) {
        var current = selectedNodes[index];
        if (current) {
            graph.clearSelection();
            for (var i = 0; i < current.length; i++) {
                current[i].color = groupColorMap[index];
            }
            graph.refresh();
        }
    };

    app.groupSelected = function (e) {
        if (graph && graph.initialized()) {
            graph.deactivateSelection();
            if (e.detail === 0) {
                graph.clearSelection();
                graph.refresh();
            } else {
                if (app.checkedGroup.length > 0) {
                    graph.clearSelection();
                    for (var i = 0; i < app.checkedGroup.length; i++) {
                        var nodes = selectedNodes[app.checkedGroup[i] - 1];
                        for (var j = 0; j < nodes.length; j++) {
                            nodes[j].color = groupColorMap[app.checkedGroup[i] - 1];
                        }
                    }
                    if (app.checkedGroup.indexOf(e.detail) < 0) {
                        groupColorNode(e.detail - 1);
                    }
                    graph.refresh();
                } else {
                    groupColorNode(e.detail - 1);
                }
                graph.activateSelection(function (err, selected) {
                    graph.clearSelection();
                    if (selected != null) {
                        selected.forEach(function (n) {
                            n.originalColor = n.color;
                            n.color = groupColorMap[e.detail - 1];
                        });
                        selectedNodes[e.detail - 1] = selected;
                    }
                    graph.refresh();

                    var nodeIndex = [];
                    var selection = selectedNodes[e.detail - 1];
                    for (var i = 0; i < selection.length; i++) {
                        nodeIndex.push(selection[i].id);
                    }
                    sendEvent('node_info', {group: e.detail, nodes: nodeIndex});
                    app.set('groups.' + (e.detail) + '.nodes', nodeIndex.length);
                });
            }
        }
    };

    var getCheckedGroup = function () {
        var groups = [];

        for (var i = 0; i < app.groups.length; i++) {
            if (app.groups[i].checked) {
                var nodeIndex = [];
                var selection = selectedNodes[app.groups[i].no - 1];
                for (var j = 0; j < selection.length; j++) {
                    nodeIndex.push(selection[j].id);
                }
                groups.push({no: app.groups[i].no, nodes: nodeIndex});
            }
        }

        return groups
    }

    app.showNodeContent = function (e) {
        sendEvent('show_node', {group: getCheckedGroup(),
            label: app.nodeLabeling});
    };

    app.runGeographer = function (e) {
        sendEvent('geography', {group: getCheckedGroup(),
            latitude: app.latitudeVar, longitude: app.longitudeVar, label: app.nodeLabeling});
    };

    app.findPoint = function (e) {
        sendEvent('find_point', {label: app.nodeLabeling, point: app.findPointLabel});
    };

    app.compare = function (e) {
        var groups = [];

        for (var i = 0; i < app.groups.length; i++) {
            if (app.groups[i].checked) {
                var nodeIndex = [];
                var selection = selectedNodes[app.groups[i].no - 1];
                for (var j = 0; j < selection.length; j++) {
                    nodeIndex.push(selection[j].id);
                }
                groups.push({no: app.groups[i].no, nodes: nodeIndex});
            }
        }

        sendEvent('compare_node', {group: groups});        
    };

    app.groupChecked = function (e) {
        var checked = [];
        for (var i = 0; i < e.detail.length; i++) {
            if (e.detail[i].checked) {
                checked.push(i);
            }
        }
        app.set('checkedGroup', checked);
    };

    app.postAnalysis = function () {
        $('#post-analysis').toggle();
    };

    app.nodeInspector = function () {
        $('#node-inspector').toggle();
    };

    app.geographer = function () {
        $('#geographer').toggle();
    };

    app.analysisResult = function () {
        $('#analysis-result').toggle();
    };

    app.postColoring = function () {
        $('#post-coloring').toggle();
    };

    app.coloringChange = function () {
        sendEvent('coloring', {'no': app.coloringVariable});
    };

    app.selectAllVariable = function (e) {
        app.selectedVariables = range(app.variables.length);
    };

    app.deselectAllVariable = function (e) {
        app.selectedVariables = [];
    };

    app.runAnalysis = function (e) {
        var selected = [];
        for (var i = 0; i < app.selectedVariables.length; i++) {
            selected.push(app.variables[app.selectedVariables[i]]);
        }
        var msg = {
            'type': 'analyze',
            'variables': selected,
            'bins': app.clusteringHistogramBins,
            'metric': app.metric
        }
        ws.send(JSON.stringify(msg));
    };

    window.addEventListener('WebComponentsReady', function () {
        //setTimeout(function () {
            $('#left-drawer').style.width = '300px';
            $('#drawer').drawerWidth = '340px';

            $('#datafile').on('change', function (e) {
                var files = e.target.files;

                if (files.length < 1) {
                    return;
                }

                var file = files[0];
                app.datasetFile = file.name;
                var reader = new FileReader();
                reader.onload = function (e) {
                    ws.send(JSON.stringify({'type': 'upload_data', 'content': e.target.result}));
                    document.title = 'Knotter - ' + app.datasetFile;
                };
                reader.readAsText(file);
            });
       // }, 1);
    });
})(document);
