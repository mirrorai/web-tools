var Editor = new function() {

    var editor = this;

    var canvas;
    var context;

    // mouse position and state
    var mousePosX;
    var mousePosY;

    var isNewRect;
    var isRectNotShow;

    var isDragging;
    var isMouseDown;
    // current selected object
    var currentSelectedObject;
    var currentObject;
    var currentHandle;

    // canvas active region
    var originX;
    var originY;
    var viewWidth;
    var viewHeight;
    var activeWidth;
    var activeHeight;
    // canvas global position
    var canvasOffsetX;
    var canvasOffsetY;
    // init flag
    var isInited = false;
    var isRemarking = false;

    // parameters
    var pad;
    var canvasId;
    var origWidth;
    var origHeight;

    // bboxes
    var origBboxes = [];
    var boxes = [];
    var origDetections = [];

    var cacheLabels = {};

    var MIN_BOX = 10;

    var disabled;

    var cellTypeId;
    var remarking;

    var currentTypeId;

    function Rectangle(x, y, width, height) {

        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;

        this.lineWidth = 3;
        this.lineWidthShadow = 2;
        this.lineWidthSelected = 4;

        this.color = "#2ac051"; // "#3cae5a"; // green
        this.colorSelected = "#30db5c"; // "#62be7a"; // light green
        this.colorDashed = "#d6af15"; // sand
        this.colorDashedSelected = "#f2c615"; // light sand
        this.colorFP = '#c61d1d';

        // constraints for position and size
        this.minWidth = MIN_BOX;
        this.minHeight = MIN_BOX;

        this.label = null;
        this.handleRadius = 3;
        this.handleRadiusSelected = 4;
        this.isSquare = false;
        this.isDragging = false;
        this.onsizechanged = null;
        this.dashed = false;
        this.moveHandleSz = 2;
        this.selected = false;
        this.cell_type_id = -1;
        this.shadow = false;

        this.active_fp = false;
        this.is_detection = false;

        this.originalBbox = null;
        this.originalDetection = null;

        this.detector_id = -1;
        this.score = 0.0;

        var self = this;

        // draw rectangle with handles
        this.draw = function(context) {

            context.lineWidth = self.lineWidth;
            if(self.selected) {
                context.lineWidth = self.lineWidthSelected;
            } else if (self.shadow) {
                context.lineWidth = self.lineWidthShadow;
            }

            // line style and color
            context.setLineDash([]);
            var color = self.color;
            if(self.dashed) {
                context.setLineDash([5, 3]);

                color = self.colorDashed;
                if(self.selected) {
                    color = self.colorDashedSelected;
                }
            } else if(self.selected) {
                color = self.colorSelected;
            }

            if(self.active_fp) {
                color = this.colorFP;
            }

            context.strokeStyle = color;
            context.strokeRect(self.x, self.y, self.width, self.height);
            context.fillStyle = color;

            if(!self.shadow) {
                self.drawHandles(context);
            }

            // draw move handle
            // var centerX = self.x + self.width / 2 - self.moveHandleSz;
            // var centerY = self.y + self.height / 2 - self.moveHandleSz;
            // context.fillRect(centerX, centerY, 2 * self.moveHandleSz, 2 * self.moveHandleSz);

            if(self.label) {
                var x = self.x + self.width - 3;
                var y = self.y + 13;
                context.font = "16px Arial";
                context.textAlign = 'right';
                var text = self.label;
                context.fillText(text, x, y);
            }
            context.setLineDash([]);
        };
        this.drawHandles = function(context) {
            var radius = self.handleRadius;
            if(self.selected) {
                radius = self.handleRadiusSelected;
            }

            //#1 top-left handle
            var centerX = self.x;
            var centerY = self.y;
            context.fillRect(centerX - radius, centerY - radius, 2 * radius, 2 * radius);

            //#2 top-right handle
            var centerX = self.x + self.width;
            var centerY = self.y;
            context.fillRect(centerX - radius, centerY - radius, 2 * radius, 2 * radius);

            //#3 bottom-right handle
            var centerX = self.x + self.width;
            var centerY = self.y + self.height;
            context.fillRect(centerX - radius, centerY - radius, 2 * radius, 2 * radius);

            //#4 bottom-left handle
            var centerX = self.x;
            var centerY = self.y + self.height;
            context.fillRect(centerX - radius, centerY - radius, 2 * radius, 2 * radius);
        };
        // check point is inside rectangle
        this.isInside = function(x, y) {
            return x >= self.x && x < self.x + self.width && y > self.y && y < self.y + self.height;
        };
        // check point is on borders
        this.isOnBorder = function(x, y) {
            var x00 = self.x - self.handleRadius;
            var x01 = self.x + self.handleRadius;
            var x10 = x00 + self.width;
            var x11 = x01 + self.width;

            var y00 = self.y - self.handleRadius;
            var y01 = self.y + self.handleRadius;
            var y10 = y00 + self.height;
            var y11 = y01 + self.height;

            if(y >= y00 && y <= y11 && x >= x00 && x <= x01) {
                return true;
            }
            if(y >= y00 && y <= y11 && x >= x10 && x <= x11) {
                return true;
            }
            if(x >= x00 && x <= x11 && y >= y00 && y <= y01) {
                return true;
            }
            if(x >= x00 && x <= x11 && y >= y10 && y <= y11) {
                return true;
            }

            return false;
        };
        // check point is over move handle
        this.isOnMoveHandle = function(x, y) {

            var centerX = self.x + self.width / 2;
            var centerY = self.y + self.height / 2;

            var x0 = centerX - self.moveHandleSz;
            var x1 = centerX + self.moveHandleSz;
            var y0 = centerY - self.moveHandleSz;
            var y1 = centerY + self.moveHandleSz;

            if(x >= x0 && x < x1 && y > y0 && y < y1) {
                return true;
            } else {
                return false;
            }
        };
        // check point over one of the handles
        this.overHandle = function(x, y) {

            var handleNum = 0;

            var x_left_max = self.x + self.handleRadius;
            var x_left_min = self.x - self.handleRadius;

            var x_right_max = self.x + self.width + self.handleRadius;
            var x_right_min = self.x + self.width - self.handleRadius;

            var y_top_max = self.y + self.handleRadius;
            var y_top_min = self.y - self.handleRadius;

            var y_bottom_max = self.y + self.height + self.handleRadius;
            var y_bottom_min = self.y + self.height - self.handleRadius;

            var isOverHandle1 = x < x_left_max && x > x_left_min;
            isOverHandle1 = isOverHandle1 && y < y_top_max;
            isOverHandle1 = isOverHandle1 && y > y_top_min;

            var isOverHandle2 = x < x_right_max && x > x_right_min;
            isOverHandle2 = isOverHandle2 && y < y_top_max;
            isOverHandle2 = isOverHandle2 && y > y_top_min;

            var isOverHandle3 = x < x_right_max && x > x_right_min;
            isOverHandle3 = isOverHandle3 && y < y_bottom_max;
            isOverHandle3 = isOverHandle3 && y > y_bottom_min;

            var isOverHandle4 = x < x_left_max && x > x_left_min;
            isOverHandle4 = isOverHandle4 && y < y_bottom_max;
            isOverHandle4 = isOverHandle4 && y > y_bottom_min;

            if(isOverHandle3) {
                return 3;
            } else if(isOverHandle1) {
                return 1;
            } else if(isOverHandle2) {
                return 2;
            } else if(isOverHandle4) {
                return 4;
            } else {
                return 0;
            }
        };
        // drag handle for specified shift
        this.dragHandle = function(handleNum, deltaX, deltaY) {

            var deltaAbsX = Math.abs(deltaX);
            var deltaAbsY = Math.abs(deltaY);

            if(self.isSquare) {
                var delta = deltaAbsX > deltaAbsY ? deltaAbsY : deltaAbsX;
                deltaX = Math.sign(deltaX) * delta;
                deltaY = Math.sign(deltaX) * delta;

                if(handleNum % 2 == 1) {
                    deltaY = deltaX;
                } else {
                    deltaY = -deltaX;
                }

            }

            // coordinates coefficients
            var coeffX = activeWidth / origWidth;
            var coeffY = activeHeight / origHeight;

            var minW = Math.max(1, Math.floor(coeffX * self.minWidth));
            var minH = Math.max(1, Math.floor(coeffY * self.minHeight));

            if(handleNum == 1) {

                var newX = self.x + deltaX;
                var newY = self.y + deltaY;
                var newW = self.width - deltaX;
                var newH = self.height - deltaY;

                if(newX < 0 && newX < self.x) return;
                if(newY < 0 && newY < self.y) return;

                if(newW < minW && newW < self.width) return;
                if(newH < minH && newH < self.height) return;

                if(newW > activeWidth && newW > self.width) return;
                if(newH > activeHeight && newH > self.height) return;

                self.x = newX;
                self.y = newY;
                self.width = newW;
                self.height = newH;

                if(self.x > activeWidth - self.width) self.x = activeWidth - self.width;
                if(self.y > activeHeight - self.height) self.y = activeHeight - self.height;

            } else if(handleNum == 2) {

                var newY = self.y + deltaY;
                var newW = self.width + deltaX;
                var newH = self.height - deltaY;

                if(newY < 0 && newY < self.y) return;
                if(self.x + newW > activeWidth && newW > self.width) return;

                if(newW < minW && newW < self.width) return;
                if(newH < minH && newH < self.height) return;

                if(newW > activeWidth && newW > self.width) return;
                if(newH > activeHeight && newH > self.height) return;

                self.y = newY;
                self.width = newW;
                self.height = newH;

            } else if(handleNum == 3) {

                var newW = self.width + deltaX;
                var newH = self.height + deltaY;

                if(self.x + newW > activeWidth && newW > self.width) return;
                if(self.y + newH > activeHeight && newH > self.height) return;

                if(newW < minW && newW < self.width) return;
                if(newH < minH && newH < self.height) return;

                if(newW > activeWidth && newW > self.width) return;
                if(newH > activeHeight && newH > self.height) return;

                self.width = newW;
                self.height = newH;

            } else if(handleNum == 4) {

                var newX = self.x + deltaX;
                var newW = self.width - deltaX;
                var newH = self.height + deltaY;

                if(newX < 0 && newX < self.x) return;
                if(self.y + newH > activeHeight && newH > self.height) return;

                if(newW < minW && newW < self.width) return;
                if(newH < minH && newH < self.height) return;

                if(newW > activeWidth && newW > self.width) return;
                if(newH > activeHeight && newH > self.height) return;

                self.x = newX;
                self.width = newW;
                self.height = newH;
            }

            if(self.sizechanged != null) {
                self.sizechanged(self);
            }
        };
        // drag whole rectangle
        this.drag = function(deltaX, deltaY) {

            self.x += deltaX;
            self.y += deltaY;

            if(self.x < 0) self.x = 0;
            if(self.y < 0) self.y = 0;

            if(self.x > activeWidth - self.width) self.x = activeWidth - self.width;
            if(self.y > activeHeight - self.height) self.y = activeHeight - self.height;

            if(self.sizechanged != null) {
                self.sizechanged(self);
            }
        };
        this.toggle_fp = function() {
            self.active_fp = !self.active_fp;
            self.shadow = !self.shadow;
        };
    }

    // factory method
    Rectangle.prototype.createFromDetection = function(detection, coeffX, coeffY) {
        var b = detection
        var x = b.x * coeffX;
        var y = b.y * coeffY;
        var w = b.w * coeffX;
        var h = b.h * coeffY;
        var rect = new Rectangle(x, y, w, h);
        rect.cell_type_id = b.class;
        rect.shadow = true;
        rect.active_fp = true;
        rect.is_detection = true;
        rect.detector_id = b.detector_id;
        rect.score = b.score;
        rect.origDetection = detection;

        if(b.ignore) {
            rect.dashed = true;
        }
        return rect;
    };
    Rectangle.prototype.createFromBbox = function(bbox, coeffX, coeffY,
        remarking, cellTypeID) {
        var b = bbox;
        var x = b.x * coeffX;
        var y = b.y * coeffY;
        var w = b.w * coeffX;
        var h = b.h * coeffY;
        var rect = new Rectangle(x, y, w, h);

        if(!b.ignore) {
            rect.cell_type_id = b.class;
        } else {
            rect.dashed = true;
            rect.cell_type_id = -1;
        }
        rect.origBbox = bbox;

        // show other type shadowed
        if(!remarking && !b.ignore) {
            rect.shadow = rect.cell_type_id != cellTypeID;
        }
        return rect;
    };

    // callback
    editor.onHoverObject = null;
    editor.onSelectObject = null;

    editor.disable = function() {
        disabled = true;
    }

    editor.enable = function() {
        disabled = false;
    }

    editor.enabled = function() {
        return !disabled;
    }

    editor.updateCurrentTypeId = function(typeID) {
        currentTypeId = typeID;
        if(currentSelectedObject) {
            currentSelectedObject.cell_type_id = typeID;
        }
    }

    // canvas id, view width and height, padding, real image size, initial bboxes
    editor.init = function(cnvsId, vWidth, vHeight, padValue) {

        isInited = true;
        disabled = true;
        isNewRect = false;
        isRectNotShow = false;
        currentSelectedObject = null;

        // set variables
        canvasId = cnvsId;
        pad = padValue;

        // default values
        isDragging = false;
        isMouseDown = false;
        mousePosY = 0;
        mousePosX = 0;

        // init canvas
        canvas = document.getElementById(canvasId);
        canvas.width =  viewWidth + 2 * pad;
        canvas.height = viewHeight + 2 * pad;
        context = canvas.getContext('2d');

        currentTypeId = 1;

        updateData(vWidth, vHeight, 0, 0, [], [], -1);
        initHandlers();
    };

    editor.setData = function(imW, imH, bboxes, dets, cellTypeID) {
        updateData(viewWidth, viewHeight, imW, imH, bboxes, dets, cellTypeID);
    };

    editor.resize = function(vWidth, vHeight) {
        updateData(vWidth, vHeight, origWidth, origHeight, origBboxes, origDetections, cellTypeId);
    };

    editor.clearData = function() {
        editor.unselectObjects();
        boxes = [];
    };

    editor.resetData = function() {
        editor.unselectObjects();
        updateData(viewWidth, viewHeight, origWidth, origHeight, origBboxes, origDetections, cellTypeId);
    };

    editor.resetObject = function(obj) {
        var idx = boxes.indexOf(obj);
        if (idx == -1) {
            return;
        }

        // coordinates coefficients
        var coeffX = activeWidth / origWidth;
        var coeffY = activeHeight / origHeight;

        if(obj.is_detection) {

            var rect = Rectangle.prototype.createFromDetection(obj.origDetection,
                coeffX, coeffY);

            boxes[idx] = rect;
        } else {

            var rect = Rectangle.prototype.createFromBbox(obj.origBbox,
                coeffX, coeffY, remarking, cellTypeId);

            boxes[idx] = rect;
        }
    };

    editor.removeObject = function(obj) {
        var idx = boxes.indexOf(obj);
        if (idx != -1) {
            boxes.splice(idx, 1);
        }
    }

    function updateData(vWidth, vHeight, imW, imH, bboxes, dets, cellTypeID) {

        cellTypeId = cellTypeID;
        remarking = cellTypeId < 0;

        origBboxes = bboxes;
        origDetections = dets;
        origWidth = imW;
        origHeight = imH;

        resizeCanvas(vWidth, vHeight);
        boxes = [];

        // coordinates coefficients
        var coeffX = activeWidth / origWidth;
        var coeffY = activeHeight / origHeight;

        for(var i = 0; i < bboxes.length; i++) {
            var rect = Rectangle.prototype.createFromBbox(bboxes[i],
                coeffX, coeffY, remarking, cellTypeId);
            if(rect) {
                boxes.push(rect);
            }
        }


        for(var i = 0; i < dets.length; i++) {
            var rect = Rectangle.prototype.createFromDetection(dets[i],
                coeffX, coeffY);
            if(rect) {
                boxes.push(rect);
            }
        }
    }

    function resizeCanvas(vWidth, vHeight) {

        if(!isInited) {
            return;
        }

        var imRatio = origWidth / origHeight;
        var vRatio = vWidth / vHeight;

        if(imRatio <= vRatio) {
            // fit height
            activeHeight = vHeight;
            activeWidth = origWidth / origHeight * activeHeight;
            originX = (vWidth - activeWidth) / 2 + pad;
            originY = pad;
        } else {
            // fit width
            activeWidth = vWidth;
            activeHeight = origHeight / origWidth * activeWidth;
            originX = pad;
            originY = (vHeight - activeHeight) / 2 + pad;
        }

        viewWidth = vWidth;
        viewHeight = vHeight;

        // update canvas
        canvas.width =  viewWidth + 2 * pad;
        canvas.height = viewHeight + 2 * pad;

        var canvasEl = $('#' + canvasId);
        canvasEl.css({
            'width': canvas.width + 'px',
            'height': canvas.height + 'px',
            'top': -pad + 'px',
            'left': -pad + 'px'
        });

        var canvasOffset = canvasEl.offset();
        canvasOffsetX = canvasOffset.left;
        canvasOffsetY = canvasOffset.top;
    };

    function selectObject(obj) {

        if(currentSelectedObject) {
            currentSelectedObject.selected = false;
        }

        if(obj) {
            obj.selected = true && remarking;
            if(obj.active_fp) {
                obj.active_fp = false;
                obj.shadow = false;
            }

            currentSelectedObject = obj;

            if(editor.onSelectObject) {
                editor.onSelectObject(currentSelectedObject);
            }

        } else {
            // unselect
            currentSelectedObject = null;
            if(editor.onSelectObject) {
                editor.onSelectObject(null);
            }
        }
    }

    editor.unselectObjects = function() {
        selectObject(null);
    };

    editor.draw = function() {

        if(!isInited) {
            return;
        }

        // coordinates coefficients
        var coeffX = activeWidth / origWidth;
        var coeffY = activeHeight / origHeight;

        context.clearRect(0, 0, canvas.width, canvas.height);

        context.translate(originX, originY);

        for(var i = 0; i < boxes.length; i++) {
            boxes[i].draw(context);
        }

        context.translate(-originX, -originY);
    };

    function initHandlers() {

        var $canvas = $(canvas);
        var clicks = 0;
        var dblclickTimeout = 200;

        $(document).on('mouseup', function(e) {

            if(disabled) {
                return;
            }

            if(e.which == 1) {

                var popupX = e.pageX;
                var popupY = e.pageY;

                // selectObject(null);
                // editor.draw();

                isNewRect = false;
                isMouseDown = false;
                isDragging = false;
            }
            // canvas.style.cursor = 'default';
        });

        $(document).on('mousemove', function(e) {

            if(disabled) {
                return;
            }

            var mousePosNewX = e.pageX - canvasOffsetX - originX;
            var mousePosNewY = e.pageY - canvasOffsetY - originY;

            if(isDragging) {
                var deltaX = mousePosNewX - mousePosX;
                var deltaY = mousePosNewY - mousePosY;

                if(currentHandle) {
                    currentObject.dragHandle(currentHandle, deltaX, deltaY);
                } else {
                    currentObject.drag(deltaX, deltaY);
                }

                editor.draw();

                if(isRectNotShow) {
                    // new Rectangle
                    var isW = currentObject.width > MIN_BOX;
                    var isH = currentObject.height > MIN_BOX;
                    if(isW && isH) {
                        currentObject.cell_type_id = currentTypeId;
                        boxes.push(currentObject);
                        // selectObject(currentObject);
                        editor.draw();
                        isRectNotShow = false;
                    }
                }
            }

            var isInsideCanvas = mousePosNewX >= 0 && mousePosNewX < activeWidth;
            isInsideCanvas = isInsideCanvas && mousePosNewY >= 0 && mousePosNewY < activeHeight;

            isInsideCanvas = isInsideCanvas && mousePosX >= 0 && mousePosX < activeWidth;
            isInsideCanvas = isInsideCanvas && mousePosY >= 0 && mousePosY < activeHeight;

            if(!isDragging && isInsideCanvas && isMouseDown) {

                //creating new rectangle
                var deltaX = mousePosNewX - mousePosX;
                var deltaY = mousePosNewY - mousePosY;
                var rectW = Math.max(deltaX, MIN_BOX);
                var rectH = Math.max(deltaY, MIN_BOX);
                var rect = new Rectangle(mousePosX, mousePosY, deltaX, deltaY);
                if(!remarking) {
                    rect.cell_type_id = cellTypeId;
                }

                var coeffX = activeWidth / origWidth;
                var coeffY = activeHeight / origHeight;

                // rect.minWidth = Math.floor(coeffX * MIN_BOX);
                // rect.minHeight = Math.floor(coeffY * MIN_BOX);

                currentObject = rect;
                currentHandle = 3;

                isDragging = true;
                isNewRect = true;
                isRectNotShow = true;

                editor.draw();
            }

            if(!isDragging && isInsideCanvas && !isMouseDown) {

                var lastObject = currentObject;

                currentObject = null;
                currentHandle = null;

                // is over handle
                var isOverHandle = false;
                for(var i = 0; i < boxes.length; i++) {

                    var skip = boxes[i].shadow && !boxes[i].active_fp;
                    if(skip) {
                        continue;
                    }

                    if(boxes[i] != currentSelectedObject) {
                        continue;
                    }

                    if((handleNum = boxes[i].overHandle(mousePosNewX, mousePosNewY))) {
                        isOverHandle = true;
                        currentObject = boxes[i];
                        currentHandle = handleNum;

                        if(currentObject == currentSelectedObject) {
                            if(currentHandle % 2 == 0) {
                                canvas.style.cursor = 'ne-resize';
                            } else {
                                canvas.style.cursor = 'se-resize';
                            }
                        }

                        break;
                    }
                }

                // is over edge
                if(!isOverHandle) {
                    var isOverRect = false;
                    var overObjects = [];
                    for(var i = 0; i < boxes.length; i++) {
                        // reverse order
                        var index = boxes.length - i - 1;
                        var skip = boxes[index].shadow && !boxes[index].active_fp;
                        if(skip) {
                            continue;
                        }

                        if(boxes[index].isInside(mousePosNewX, mousePosNewY)) {
                            isOverRect = true;
                            overObjects.push(boxes[index]);
                        }
                    }
                    if(!isOverRect) {
                        canvas.style.cursor = 'default';
                    } else {
                        currentObject = overObjects[0];
                        var minArea = currentObject.width * currentObject.height;
                        for(var i = 1; i < overObjects.length; i++) {
                            var area = overObjects[i].width * overObjects[i].height;
                            if(minArea > area) {
                                currentObject = overObjects[i];
                                minArea = area;
                            }
                        }

                        if(currentSelectedObject == currentObject) {
                            canvas.style.cursor = 'move';
                        }
                    }
                }

                // update tooltip
                if(remarking) {
                    if(lastObject != currentObject || currentObject) {
                        if(currentObject && !currentObject.dashed) {
                            var type_id = currentObject.cell_type_id;
                            editor.onHoverObject(type_id, e.pageX, e.pageY);
                        } else if (currentObject == null) {
                            editor.onHoverObject(null, 0, 0);
                        }
                    }
                }
            }

            mousePosX = mousePosNewX;
            mousePosY = mousePosNewY;
        });

        $canvas.on('mousedown', function(e) {

            if(disabled) {
                return;
            }

            if(e.which == 1) {

                isMouseDown = true;
                mousePosX = e.pageX - canvasOffsetX - originX;
                mousePosY = e.pageY - canvasOffsetY - originY;

                if(currentObject && currentSelectedObject == currentObject) {
                    isDragging = true;
                    // selectObject(currentObject);
                    editor.draw();
                } else {
                    editor.unselectObjects();
                }
                editor.onHoverObject(null, 0, 0);

                isNewRect = false;
                isRectNotShow = false;
            }
        });

        $('#' + canvasId).on('contextmenu', function(e) {

            if(disabled) {
                return;
            }

            if(currentObject != null) {
                clicks++;
                if (clicks == 1) {
                    // single click
                    if(currentObject != null) {
                        // update current object
                        selectObject(currentObject);
                        editor.draw();
                    }
                    setTimeout(function(){
                        if(clicks == 1) {

                        } else {
                            // double click
                            // remove current object
                            var idx = boxes.indexOf(currentObject);
                            if (idx != -1) {
                                boxes.splice(idx, 1);
                            }
                            currentObject = null;
                            selectObject(currentObject);
                            editor.draw();
                        }
                        clicks = 0;
                    }, dblclickTimeout);
                }
            } else {
                selectObject(null);
                editor.draw();
            }

            editor.onHoverObject(null, 0, 0);

            return false;
        });
    }

    editor.getBoxes = function() {

        // coordinates coefficients
        var coeffX = activeWidth / origWidth;
        var coeffY = activeHeight / origHeight;

        var sendBoxes = [];
        for(var i = 0; i < boxes.length; i++) {
            var b = boxes[i]
            if(b.shadow) {
                continue;
            }

            var box = new Object();
            box.x = Math.round(b.x / coeffX);
            box.y = Math.round(b.y / coeffY);
            box.w = Math.round(b.width / coeffX);
            box.h = Math.round(b.height / coeffY);
            box.ignore = b.dashed;
            box.class = b.cell_type_id;
            sendBoxes.push(box);
        }
        return sendBoxes;
    }

    editor.getDetections = function() {

        // coordinates coefficients
        var coeffX = activeWidth / origWidth;
        var coeffY = activeHeight / origHeight;

        var sendBoxes = [];
        for(var i = 0; i < boxes.length; i++) {
            var b = boxes[i]
            if(!b.active_fp) {
                continue;
            }

            var box = new Object();
            box.x = Math.round(b.x / coeffX);
            box.y = Math.round(b.y / coeffY);
            box.w = Math.round(b.width / coeffX);
            box.h = Math.round(b.height / coeffY);
            box.class = b.cell_type_id;
            box.detector_id = b.detector_id;
            box.score = b.score;
            sendBoxes.push(box);
        }
        return sendBoxes;
    }
};