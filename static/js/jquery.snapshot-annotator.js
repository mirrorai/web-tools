// Look at this for structure of jQuery plugin
// https://alexsexton.com/blog/2010/02/using-inheritance-patterns-to-organize-large-jquery-applications/

(function () {
    "use strict";

    var dotDotDistance = function (p0, p1) {
            var x0 = p0.x, y0 = p0.y, x1 = p1.x, y1 = p1.y;
            return Math.sqrt(Math.pow(x0 - x1, 2) + Math.pow(y0 - y1, 2));
        },

        scalarProd = function (v0, v1) {
            return v0.x * v1.x + v0.y * v1.y;
        },

        dotSegmentDistance = function (p, start, end) {
            var x = p.x,
                y = p.y,
                x0 = start.x,
                y0 = start.y,
                x1 = end.x,
                y1 = end.y,
                a = y1 - y0,
                b = x0 - x1,
                c = -(a * x0 + b * y0);

            if (scalarProd({x: x - x0, y: y - y0}, {x: x1 - x0, y: y1 - y0}) <= 0 ||
                scalarProd({x: x - x1, y: y - y1}, {x: x0 - x1, y: y0 - y1}) <= 0) {
                return Math.min(dotDotDistance(p, start), dotDotDistance(p, end));
            }

            return Math.abs((a * x + b * y + c) / Math.sqrt(a * a + b * b));
        },

        pointInPolygon = function (point, vs) {
            // ray-casting algorithm based on
            // http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html

            var x = point.x, y = point.y;

            var inside = false;
            for (var i = 0, j = vs.length - 1; i < vs.length; j = i++) {
                var xi = vs[i].x, yi = vs[i].y;
                var xj = vs[j].x, yj = vs[j].y;

                var intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
                if (intersect) {
                    inside = !inside;
                }
            }

            return inside;
        },

        Annotator = {
            // TODO: use canvas transformation instead of scaleX, scaleY and implement image zooming and moving
            ACCEPT_CLICK_DISTANCE: 10,
            BBOXES_SIZE: 8,
            FONT_SIZE: 30,

            canvasPoint: function(e) {
                // Return position of cursor on canvas
                return {
                    x: e.pageX - $(e.target).offset().left,
                    y: e.pageY - $(e.target).offset().top
                };
            },

            canvasToImage: function (p) {
                // Converts point from canvas coordinates to image coordinates
                return {
                    x: p.x / this.scaleX,
                    y: p.y / this.scaleY
                };
            },

            imageToCanvas: function (p) {
                // Converts point from image coordinates to canvas coordinates
                return {
                    x: p.x * this.scaleX,
                    y: p.y * this.scaleY
                };
            },

            setActiveFigure: function(newActiveFigure) {
                if (this.activeFigure !== newActiveFigure) {
                    this.activeFigure = newActiveFigure;
                    if (this.options.onSelectionChanged !== null) {
                        this.options.onSelectionChanged(newActiveFigure);
                    }
                }
            },

            move: function (e) {
                this.figures[this.activeFigure].points[this.activePoint] = this.canvasToImage(this.canvasPoint(e));
                this.redraw();
            },

            stopDrag: function () {
                $(this.canvas).off('mousemove');
                this.activePoint = null;
            },

            nearestPoint: function (imagePoint) {
                // Finds nearest point among all figures
                var bestFigureId = null, bestPointId = null, bestDist = 1e20;
                this.figures.forEach(function (figure, figureId) {
                    figure.points.forEach(function (point, pointId) {
                        var dist = dotDotDistance(imagePoint, point);
                        if (dist < bestDist) {
                            bestFigureId = figureId;
                            bestPointId = pointId;
                            bestDist = dist;
                        }
                    });
                });
                return {
                    figureId: bestFigureId,
                    pointId: bestPointId,
                    distance: bestDist
                };
            },

            nearestSegment: function (imagePoint) {
                // Finds nearest segment among all figures
                var bestFigureId = null, bestPointId = null, bestDist = 1e20;
                this.figures.forEach(function (figure, figureId) {
                    var startId, endId, dist;
                    for (startId = 0; startId < figure.points.length; startId += 1) {
                        endId = (startId + 1) % figure.points.length;
                        dist = dotSegmentDistance(imagePoint, figure.points[startId], figure.points[endId]);
                        if (dist < bestDist) {
                            bestFigureId = figureId;
                            bestPointId = startId;
                            bestDist = dist;
                        }
                    }
                });
                return {
                    figureId: bestFigureId,
                    pointId: bestPointId,
                    distance: bestDist
                };
            },

            boundingFigure: function(imagePoint) {
                var result = null;
                this.figures.every(function (figure, figureId) {
                    if (pointInPolygon(imagePoint, figure.points)) {
                        result = figureId;
                        return false;
                    }
                    return true;
                });
                return result;
            },

            rightClick: function(e) {
                // If clicking:
                //     on point -- remove it
                //     else -- deselect current figure
                var p, res;
                e.preventDefault();
                p = this.canvasPoint(e);
                res = this.nearestPoint(this.canvasToImage(p));
                if (res.figureId !== null && res.distance < this.ACCEPT_CLICK_DISTANCE) {
                    this.figures[res.figureId].points.splice(res.pointId, 1);
                    if (this.figures[res.figureId].points.length === 0) {
                        this.figures.splice(res.figureId, 1);
                        this.setActiveFigure(null);
                    } else {
                        this.setActiveFigure(res.figureId);
                    }
                    this.activePoint = null;
                    this.redraw();
                    return false;
                }
                if (this.options.allowDeselect) {
                    this.setActiveFigure(null);
                }
                this.redraw();
                return false;
            },

            mouseDown: function (e) {
                // If clicking:
                //     on point -- start moving it
                //     on segment -- add point
                //     on empty space and have activeFigure -- add point to the end
                //     on empty space, no activeFigure -- try to select active figure
                var p, res;
                if (e.which === 3) {
                    return false;
                }

                e.preventDefault();

                p = this.canvasPoint(e);

                if (!this.options.readOnly) {
                    res = this.nearestPoint(this.canvasToImage(p));
                    if (res.figureId !== null && res.distance < this.ACCEPT_CLICK_DISTANCE) { // Moving point
                        this.setActiveFigure(res.figureId);
                        this.activePoint = res.pointId;
                        $(this.canvas).on('mousemove', this.move.bind(this));
                        this.redraw();
                        return false;
                    }

                    res = this.nearestSegment(this.canvasToImage(p));
                    if (res.figureId !== null && res.distance < this.ACCEPT_CLICK_DISTANCE) { // Creating point on segment
                        this.setActiveFigure(res.figureId);
                        this.activePoint = res.pointId + 1;
                        this.figures[res.figureId].points.splice(res.pointId + 1, 0, this.canvasToImage(p));
                        $(this.canvas).on('mousemove', this.move.bind(this));
                        this.redraw();
                        return false;
                    }

                    if (this.activeFigure !== null) { // Adding point to figure
                        this.activePoint = this.figures[this.activeFigure].points.length;
                        this.figures[this.activeFigure].points.push(this.canvasToImage(p));
                        $(this.canvas).on('mousemove', this.move.bind(this));
                        this.redraw();
                        return false;
                    }
                }

                if (!this.options.readOnly && this.options.allowCreatingFigures &&
                    this.figures.length < this.options.maxFigures) {
                    // Creating new figure
                    this.setActiveFigure(this.figures.length);
                    this.activePoint = null;
                    this.figures.push({
                        name: 'New figure ' + this.figures.length,
                        points: [this.canvasToImage(p)]
                    });
                } else {
                    // Move selection
                    if (this.options.allowSelectionWithClick) {
                        res = this.boundingFigure(this.canvasToImage(p));
                        if (res !== null) {
                            this.setActiveFigure(res);
                        } else if (this.options.allowDeselect) { // Don't allow to deselect
                            this.setActiveFigure(null);
                        }
                    } else {
                        this.setActiveFigure(null);
                    }
                }

                $(this.canvas).off('mousemove');
                this.redraw();
                return false;
            },

            drawFigure: function (context, figure, figureId) {
                function moreRightTop(p1, p2) {
                    if (p1.x !== p2.x) {
                        return p1.x > p2.x ? p1 : p2;
                    }
                    return p1.y < p2.y ? p1 : p2;
                }

                var i, curPoint, startPoint, rightTop, size = this.BBOXES_SIZE, figure_color;
                figure_color = figure.color || 'rgb(255, 0, 0)';
                context.strokeStyle = figure_color;
                context.lineWidth = 2;

                context.beginPath();
                startPoint = this.imageToCanvas(figure.points[0]);
                context.moveTo(startPoint.x, startPoint.y);
                for (i = 0; i < figure.points.length; i += 1) {
                    curPoint = this.imageToCanvas(figure.points[i]);
                    if (!this.options.readOnly) {
                        if (i !== 0 && i !== figure.points.length - 1) {
                            context.fillStyle = 'rgb(255, 255, 255)';
                        } else {
                            context.fillStyle = 'rgb(255, 255, 0)';
                        }
                        context.fillRect(curPoint.x - size / 2, curPoint.y - size / 2, size, size);
                        context.strokeRect(curPoint.x - size / 2, curPoint.y - size / 2, size, size);
                    }
                    if (i >= 1) {
                        context.lineTo(curPoint.x, curPoint.y);
                    }
                }
                context.closePath();
                if (figureId === this.activeFigure) {
                    context.fillStyle = (figure_color.substr(0, figure_color.length - 1) + ', 0.3)').replace('rgb', 'rgba');
                    context.fill();
                }
                context.stroke();

                if (this.options.drawNames) {
                    rightTop = this.imageToCanvas(figure.points.reduce(moreRightTop));
                    context.font = Math.round(this.FONT_SIZE * this.scaleY) + 'px Courier';
                    context.fillStyle = figure.color || 'rgb(255, 20, 20)';
                    context.fillText(figure.name, rightTop.x - 80 * this.scaleX, rightTop.y + 30 * this.scaleY);
                }
            },

            drawShade: function (context) {
                // http://stackoverflow.com/questions/6271419/how-to-fill-the-opposite-shape-on-canvas
                var viewportWidth = this.canvas.width,
                    viewportHeight = this.canvas.height,
                    shadingContext = this.shadingCanvas.getContext('2d'),
                    curPoint, i;
                if (this.options.shadingPolygon !== null) {
                    // Ensure same dimensions
                    $(this.shadingCanvas).attr('width', viewportWidth);
                    $(this.shadingCanvas).attr('height', viewportHeight);

                    // Fill with grey color for further multiplication
                    shadingContext.fillStyle = 'rgb(128, 128, 128)';
                    shadingContext.fillRect(0, 0, viewportWidth, viewportHeight);

                    // Draw the shape you want to take out
                    shadingContext.globalCompositeOperation = 'xor';
                    shadingContext.beginPath();
                    curPoint = this.imageToCanvas(this.options.shadingPolygon[0]);
                    shadingContext.moveTo(curPoint.x, curPoint.y);
                    for (i = 1; i < this.options.shadingPolygon.length; i += 1) {
                        curPoint = this.imageToCanvas(this.options.shadingPolygon[i]);
                        shadingContext.lineTo(curPoint.x, curPoint.y);
                    }
                    shadingContext.closePath();
                    shadingContext.fill();

                    // Draw mask on the image
                    context.globalCompositeOperation = 'multiply';
                    context.drawImage(this.shadingCanvas, 0, 0);
                    context.globalCompositeOperation = 'source-over';
                }
            },

            redraw: function () {
                if (!this.image.complete) {
                    return;
                }
                var viewportWidth = $(this.container).width(),
                    viewportHeight = parseInt(viewportWidth * 1.0 / this.image.width * this.image.height),
                    context = this.canvas.getContext('2d');
                $(this.canvas)
                    .attr('width', viewportWidth)
                    .attr('height', viewportHeight);
                this.scaleX = viewportWidth * 1.0 / (this.options.overrideImageWidth || this.image.width);
                this.scaleY = viewportHeight * 1.0 / (this.options.overrideImageHeight || this.image.height);

                // Clear canvas
                context.clearRect(0, 0, viewportWidth, viewportHeight);

                // Draw snapshot
                context.drawImage(
                    this.image,
                    0,
                    0,
                    this.image.width,
                    this.image.height,
                    0,
                    0,
                    viewportWidth,
                    viewportHeight
                );

                // Draw shade overlay
                this.drawShade(context);

                this.figures.forEach( function (figure, figureId) {
                    this.drawFigure(context, figure, figureId);
                }.bind(this));

                if (this.options.onRedraw !== null) {
                    this.options.onRedraw();
                }
            },

            destroy: function () {
                $(this.container).empty();
            },

            defaultOptions: {
                figures: [],
                allowCreatingFigures: true,
                readOnly: false,
                maxFigures: 100500,
                allowZeroAreaFigures: false,
                allowSelfIntersections: false,
                allowDeselect: true,
                onRedraw: null,
                autoSelectFirst: true,
                shadingPolygon: null,
                overrideImageWidth: null,
                overrideImageHeight: null,
                drawNames: false,
                onSelectionChanged: null,
                allowSelectionWithClick: true
            },

            init: function (options, container) {
                // Setup options
                //noinspection NodeModulesDependencies
                this.options = $.extend({}, this.defaultOptions, options);

                // Setup DOM elements
                this.container = container;
                this.canvas = document.createElement('canvas');
                // http://stackoverflow.com/questions/5804256/image-inside-div-has-extra-space-below-the-image
                // Need following line to exclude 5 pixel gap below canvas
                this.canvas.style.display = 'block';
                this.shadingCanvas = document.createElement('canvas');

                // Alias figures from options for easier access
                this.figures = this.options.figures;

                // Setup initial selection
                this.setActiveFigure(null);
                if (this.options.autoSelectFirst && this.figures.length > 0) {
                    this.setActiveFigure(0);
                }
                this.activePoint = null;

                // Create image
                this.image = new Image();
                this.image.src = options.imageUrl;
                $(this.image).load(this.redraw.bind(this));

                // Append elements to DOM
                $(this.container).empty();
                $(this.container).append(this.canvas);
                $(this.canvas)
                    .on('mousedown', this.mouseDown.bind(this))
                    .on('contextmenu', this.rightClick.bind(this))
                    .on('mouseup', this.stopDrag.bind(this));

                $(window).resize(this.redraw.bind(this));

                return this;
            }
        };

    if (typeof Object.create !== 'function') {
        Object.create = function (o) {
            function F() {}
            F.prototype = o;
            return new F();
        };
    }

    (function($){
        $.fn.annotator = function(options) {
            return this.each(function (){
                var myAnnotator = Object.create(Annotator);
                myAnnotator.init(options, this);
                $.data(this, 'annotator', myAnnotator);
            });
        };
    })(jQuery);
}());
