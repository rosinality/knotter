(function() {
'use strict';

	if (typeof sigma === 'undefined')
		throw 'sigma is not declared';

	sigma.utils.pkg('sigma.plugins');

	var startPosition;
	var WIDTH;
	var HEIGHT;
	var ctx;
	var gCanvas;
	var graph;
	var callback;
	
	var is_selecting;

	function mousedown(e){
        if (e.which === 3 || e.button === 2) {
	    	startPosition={x:e.layerX, y:e.layerY};
	    	gCanvas.addEventListener('mousemove',mousemove);
	    	is_selecting = true;
        }
	}

	function mouseup(e){
        if (is_selecting) {
		gCanvas.removeEventListener('mousemove',mousemove);
	//	gCanvas.removeEventListener('mousedown',mousedown);
	//	gCanvas.removeEventListener('mouseup',mouseup);
		is_selecting = false;
		clear(ctx);
	//	graph.settings('mouseEnabled', true);

		callback(null, getNodesInArea(startPosition.x, startPosition.y, e.layerX, e.layerY));
        }
	}

	function getNodesInArea(x1, y1, x2, y2){
		var nodesInArea = [];
		var startX;
		var endX;
		var startY;
		var endY;

		if (x1 > x2){
			startX = x2;
			endX = x1;
		}else{
			startX = x1;
			endX = x2;
		}

		if (y1 > y2){
			startY = y2;
			endY = y1;
		}else{
			startY = y1;
			endY = y2;
		}
		
		graph.camera.quadtree._cache.result.forEach(function(node){
			var nodeX = node['cam0:x'];
			var nodeY = node['cam0:y'];
			//console.log(node);
			if ((nodeX > startX) && (nodeX < endX) && 
							(nodeY > startY) && (nodeY < endY)){
				nodesInArea.push(node);
			}
		});
	//	console.log(nodesInArea);
		return nodesInArea;
	}

	function mousemove(e){
		if (!is_selecting) {
			return;
		}
		clear(ctx);
		ctx.beginPath();
		ctx.lineWidth='1';
		ctx.setLineDash([6]);
		ctx.strokeStyle='black';
		ctx.rect(startPosition.x, startPosition.y, e.layerX - startPosition.x, e.layerY - startPosition.y); 
		ctx.stroke();
	}

	sigma.plugins.activateMouseEvents = function(s, cb) {
		if (!s){
			cb('graph not supplied');
		}else{
			var renderer = s.renderers[0];
			var container = renderer.container;
			graph = s;
			callback = cb;
		//	graph.settings('mouseEnabled', false);
			gCanvas = container.lastChild;
			HEIGHT = gCanvas.height;
	  		WIDTH = gCanvas.width;
			ctx = gCanvas.getContext('2d');
			gCanvas.addEventListener('mousedown',mousedown);
			gCanvas.addEventListener('mouseup',mouseup);
            gCanvas.addEventListener('contextmenu', function (e) {
                e.preventDefault();
                return false;
            });
		}
	};

    sigma.plugins.deactivateMouseEvents = function () {
        if (gCanvas) {
            gCanvas.removeEventListener('mousedown', mousedown);
            gCanvas.removeEventListener('mouseup', mouseup);
        }
    };

	function clear(c) {
		c.clearRect(0, 0, WIDTH, HEIGHT);
	}
}).call(window);
