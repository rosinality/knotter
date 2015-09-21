var d3plot = (function () { 
    var functions = {
        scale: null,
        
        makeCubeHelix: function (min, center, max) {
            return d3.scale.cubehelix()
                .domain([min, center, max])
                .range([
                        d3.hsl(-20, 0.75, 0.35),
                        d3.hsl(  80, 1.50, 0.80),
                        d3.hsl( 230, 0.75, 0.35)
                ]);
        },

        makeHistogram: function (target, values, bins) {
            //var values = d3.range(1000).map(d3.random.normal(0, 1));
            var formatCount = d3.format(",.0f");

            var margin = {top: 10, right: 30, bottom: 30, left: 30},
            width = 300 - margin.left - margin.right,
            height = 120 - margin.top - margin.bottom;

            var adjuster = function (value, increase) {
                if (value <= 0) {
                    if (increase) {
                        return value * 0.8;
                    } else {
                        return value * 1.2;
                    }
                } else {
                    if (increase) {
                        return value * 1.2;
                    } else {
                        return value * 0.8
                    }
                }
            }

            var min = Math.min.apply(Math, values);
            var max = Math.max.apply(Math, values);

            var oneHalfTick = (max - min) / bins * 1.5;

            var x = d3.scale.linear()
                .domain([min, max + oneHalfTick])
                .range([0, width]);
            var center = (max + min) / 2;
            var scale = this.makeCubeHelix(min, center, max);
            this.scale = scale;
            var data = d3.layout.histogram()
                .bins(x.ticks(bins))
                (values);

            var y = d3.scale.linear()
                .domain([0, d3.max(data, function(d) { return d.y; })])
                .range([height, 0]);

            var xAxis = d3.svg.axis()
                .scale(x)
                .tickValues([x.domain()[0], max])
                .orient("bottom");

            var svg = d3.select(target)
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

            var bar = svg.selectAll(".bar")
                .data(data)
                .enter().append("g")
                .attr("class", "bar")
                .attr('fill',
                        function (d) { return scale(d.x); })
                .attr("transform", function(d) {
                    return "translate(" + x(d.x) + "," + y(d.y) + ")";
                });

            bar.append("rect")
                .attr("x", 1)
                .attr('width', .5)
                .attr("width", x(data[1].x) - x(data[0].x) - 1)
                .attr("height", function(d) { return height - y(d.y); });

            svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + height + ")")
                .call(xAxis);
        }
    };

    return functions;
})();
