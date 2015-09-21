var geography = (function () {
    var map = null;

    return {
        createMap: function (target, latitude, longitude) {
            if (map != null) {
                map.parentNode.removeChild(map);
                map = null;
            }
            map = document.createElement('google-map');
            map.latitude = latitude;
            map.longitude = longitude;
            target.innerHTML = '';
            target.appendChild(map);
        },

        ready: function (listener) {
            map.on('google-map-ready', listener);
        },

        makeMarker: function (data) {
            var overlay = new google.maps.OverlayView();
            overlay.onAdd = function () {
                var layer = d3.select(this.getPanes().overlayLayer).append('div')
                    .attr('class', 'stations');
                
                overlay.draw = function () {
                    var projection = this.getProjection(),
                        padding = 10;

                    var transform = function (d) {
                            d = new google.maps.LatLng(d.value[0], d.value[1]);
                            d = projection.fromLatLngToDivPixel(d);
                            return d3.select(this)
                                .style('left', (d.x - padding) + 'px')
                                .style('top', (d.y - padding) + 'px');
                        };

                    var marker = layer.selectAll('svg')
                        .data(d3.entries(data))
                        .each(transform)
                        .enter().append('svg:svg')
                        .each(transform)
                        .attr('class', 'marker');
                    marker.append('svg:circle')
                        .attr('r', 4.5)
                        .attr('cx', padding)
                        .attr('cy', padding)
                        .style('fill', function (d) {
                            return groupColorMap[d.value[2]];
                        });
                    marker.append('svg:text')
                        .attr('x', padding + 7)
                        .attr('y', padding)
                        .attr('dy', '.31em')
                        .style('font-size', '12px')
                        .text(function (d) { return d.value[3]; });
                };
            };
            overlay.setMap(map.map);
        }
    };
})();
