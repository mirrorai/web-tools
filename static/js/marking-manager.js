
var MarkingManager =  new function() {
	IMAGE_SZ = 720;
	var self = this;

	var EDITOR_ID = 'editor';
	var DEFAULT_TYPE_ID = 1;

	// entry point
	self.init = function(data, is_empty) {

		initElements();
		initHandlers();

		var viewW = self.container.width();
		var viewH = self.container.height();

		self.isNextLoading = false;
		self.isSending = false;
		self.cacheLabels = {};
		self.cacheBatchLabels = {};
		self.attemptsToCloseModal = 0;
		self.do_skip = false;
		self.batch_id = null;

		// init canvas editor
		Editor.init(EDITOR_ID, viewW, viewH, 4);
		Editor.onHoverObject = updateTooltipInfo;
		Editor.onSelectObject = updatePropertyInfo;
		Editor.updateCurrentTypeId(DEFAULT_TYPE_ID);
		setTypeToPropertyBtn(DEFAULT_TYPE_ID);

		if(is_empty) {
			onEmptyData(data);
		} else {
			onDataLoaded(data);
		}
	}

	function batchId2Label(batchId) {
		if(batchId in self.cacheBatchLabels) {
            return self.cacheBatchLabels[batchId];
        } else {
        	var pickBtn = self.modalBatch.find('.type-pick-btn[batch-id='+ batchId.toString() +']');
        	if(!pickBtn) {
        		return null;
        	}

        	var label = pickBtn.text();
        	self.cacheBatchLabels[batchId] = label;
        	return label;
        }
	}

	function cellTypeId2Label(cellTypeID) {

        if(cellTypeID in self.cacheLabels) {
            return self.cacheLabels[cellTypeID];
        } else {
        	var pickBtn = self.modal.find('.type-pick-btn[data-id='+ cellTypeID.toString() +']');
        	if(!pickBtn) {
        		return null;
        	}

        	var headElement = pickBtn.closest('.panel').find('.panel-title > a');
        	if(!headElement) {
	            var label = pickBtn.text();
	            self.cacheLabels[cellTypeID] = label;
	            return label;
        	} else {
        		var label = headElement.text() + ' / ' + pickBtn.text();
        		self.cacheLabels[cellTypeID] = label;
	            return label;
        	}
        }
    }

	function updatePropertyInfo(selectedObject) {

		if(selectedObject == null) {
			self.rectResetItem.hide();
			return;
		}

		self.rectResetItem.show();
		setTypeToPropertyBtn(selectedObject.cell_type_id);

		self.rectResetItem.show();
		self.rectResetBtn.off('click');
		self.rectResetBtn.on('click', function(e) {

			e.preventDefault();
			this.blur();

			Editor.removeObject(selectedObject);
			Editor.unselectObjects();
			Editor.draw();
		});
	}

	function updateTooltipInfo(cellTypeId, pageX, pageY) {
		if(cellTypeId) {
			var label = cellTypeId2Label(cellTypeId);
			self.typeTooltip.attr('title', label);
			self.typeTooltip.css({top: pageY - 2, left: pageX + 5 });
			self.typeTooltip.tooltip('fixTitle').tooltip('show');
		} else {
			self.typeTooltip.tooltip('hide');
		}
	}

	// init UI variables
	function initElements() {
		self.imgMain = $('#img-main');
		self.container = $('#img-container');
		self.loadingText = $('#loading-text');
		self.emptyText = $('#empty-text');
		self.canvasEl = $('#editor');
		self.sampleTagLabel = $('#sample-tag-sidebar');
		self.modal = $('#popup-select-cell-type');
		self.modalBatch = $('#popup-select-batch');
		self.typeTooltip = $('#type-tooltip');
		self.batchBtn = $('#batch-select-btn');
		self.bboxInfo = $('#bbox-info');
		self.bboxInfoType = self.bboxInfo.find('#current-bbox-type');
		self.nextBtn = $('#next-btn');
		self.nextOtherBtn = $('#next-other-btn');
		self.backBtn = $('#back-btn');
		self.cytoBtn = $('#cyto-btn');
		self.smallResolBtn = $('#small-resol-btn');
		self.skipBtn = $('#skip-btn');
		self.hardBtn = $('#hard-btn');
		self.badBtn = $('#bad-btn');
		self.clearBtn = $('#clear-btn');
		self.countText = $('#count-text');
		self.countBatchText = $('#count-batch-text');
		self.rectResetBtn = $('#rect-reset-btn');
		self.rectResetItem = $('#item-rect-reset');
		self.resetBtnItem = $('#item-reset-btn');
		self.selectTypeBtns = $('.cell-type-btn');
		self.badText = $('#bad-text');
	}

	function updateHistory(historyData) {

		self.prev_id = historyData.prev_id;
		self.next_id = historyData.next_id;

		if(historyData.prev_id > 0) {
			self.backBtn.removeClass('disabled');
		} else {
			self.backBtn.addClass('disabled');
		}
	}

	function openModal(nameId) {
		// show selected section and hide other
		self.modal.find('.section-hc').hide();
		var section = self.modal.find('#section-' + nameId);
		if(section.length == 0) {
			return;
		}
		section.show();

		// reset selection inside modal
		resetModalSelection(true);
		// show modal
        self.modal.modal('show');
        // save state of editor
        self.savedEditorState = Editor.enabled();
        Editor.disable();

        // set type as selected inside modal
		var subsections = section.find('.panel-collapse');
        var cell_type_id_str = self.bboxInfoType.attr('data-id');
        var type_not_set = true;
        if(cell_type_id_str) {
        	var type_elem = section.find('.type-pick-btn[data-id="' + cell_type_id_str +'"]');
        	if(type_elem.length) {
        		var cell_type_id = parseInt(cell_type_id_str);
        		setTypeToModal(cell_type_id);
        		type_not_set = false;
        	}
        }

        // collapse subsection if there is only one subsection
        if(type_not_set && subsections.length == 1) {
        	subsections.collapse('show');
        }
    }

    function openModalBatch(batchId) {

    	if(batchId && batchId > 0) {
    		self.modalBatch.find('.type-pick-btn.active').removeClass('active');
    		self.modalBatch.find('.type-pick-btn[batch-id=' + batchId +']').addClass('active');
    	}

		// show modal
        self.modalBatch.modal('show');
        // save state of editor
        self.savedEditorState = Editor.enabled();
        Editor.disable();
    }

    function onModalClosed(e) {
    	var opt = self.modal.find('.type-pick-btn.active');

        // if type is not selected but try to close
        if(opt.length == 0 && self.attemptsToCloseModal == 0) {
            // type not selected, first attempt
            self.attemptsToCloseModal += 1;
            showWarning(self.modal);
            e.preventDefault();
            e.stopImmediatePropagation();
            return false;
        } else if (opt.length == 0) {
            // select something
            hideWarning(self.modal);
            setTypeToPropertyBtn(DEFAULT_TYPE_ID);
            Editor.updateCurrentTypeId(DEFAULT_TYPE_ID);
            return true;
        }
        self.attemptsToCloseModal = 0;

        // update data
        var typeId = parseInt(opt.attr('data-id'));
        setTypeToPropertyBtn(typeId);

        hideWarning(self.modal);
        if(self.savedEditorState) {
        	Editor.enable();
        }

        Editor.updateCurrentTypeId(typeId);
        return true;
    }

    function onModalBatchClosed(e) {
    	var opt = self.modalBatch.find('.type-pick-btn.active');

        // if type is not selected but try to close
        if(opt.length == 0 && self.attemptsToCloseModal == 0) {
            // type not selected, first attempt
            self.attemptsToCloseModal += 1;
            showWarning(self.modalBatch);
            e.preventDefault();
            e.stopImmediatePropagation();
            return false;
        } else if (opt.length == 0) {
            // select something
            hideWarning(self.modalBatch);
            return true;
        }
        self.attemptsToCloseModal = 0;

        // update data
        var batchId = parseInt(opt.attr('batch-id'));
        self.batch_id = batchId;
        setLabelToBatchBtn(self.batch_id);

        hideWarning(self.modalBatch);
        if(self.savedEditorState) {
        	Editor.enable();
        }

        sendAndRequest(self.batch_id);

        return true;
    }

    function resetModalSelection(resetToggle) {
        self.modal.find('.type-pick-btn.active').removeClass('active');
        if(resetToggle) {
             self.modal.find('.panel-collapse.in').collapse('hide');
        }
    }

    function setTypeToModal(cellTypeId) {
        var dataId = cellTypeId.toString();
        var selEl = self.modal.find('.type-pick-btn[data-id="' + dataId +'"]');
        selEl.addClass('active');
        var parent = selEl.closest('.panel-collapse');
        parent.collapse('show');
    }

    function setTypeToPropertyBtn(cellTypeId) {
    	self.bboxInfoType.attr('data-id', cellTypeId);
    	var label = cellTypeId2Label(cellTypeId);
		self.bboxInfoType.text(label);
    }

    function setLabelToBatchBtn(batchId) {
    	var label = batchId2Label(batchId);
    	self.batchBtn.text(label);
    }

    function setBad(is_bad) {
    	if(is_bad) {
    		self.badBtn.addClass('picked');
    	} else {
    		self.badBtn.removeClass('picked');
    	}
    	self.is_bad = is_bad;
    }

    function setHard(is_hard) {
    	if(is_hard) {
    		self.hardBtn.addClass('checked');
    	} else {
    		self.hardBtn.removeClass('checked');
    	}
    	self.is_hard = is_hard;
    }

    function setSmallResol(is_small_resol) {
    	if(is_small_resol) {
    		self.smallResolBtn.addClass('checked');
    	} else {
    		self.smallResolBtn.removeClass('checked');
    	}
    	self.is_small_resol = is_small_resol;
    }

    function setCyto(is_cyto) {
    	if(is_cyto) {
    		self.cytoBtn.addClass('checked');
    	} else {
    		self.cytoBtn.removeClass('checked');
    	}
    	self.is_cyto = is_cyto;
    }

    function showWarning(modal) {
        modal.find('.type-not-selected-label').show();
    }

    function hideWarning(modal) {
        modal.find('.type-not-selected-label').hide();
    }

	// init handlers
	function initHandlers() {


		self.modal.on('hide.bs.modal', function (e) {
            return onModalClosed(e);
        });

        self.modalBatch.on('hide.bs.modal', function (e) {
            return onModalBatchClosed(e);
        });

        $('.dialog-cell-type .type-pick-btn').click(function(e) {
            self.modal.find('.type-pick-btn.active').removeClass('active');
            $(this).addClass('active');
            self.modal.modal('hide');
        });

        $('.dialog-batch .type-pick-btn').click(function(e) {
            self.modalBatch.find('.type-pick-btn.active').removeClass('active');
            $(this).addClass('active');
            self.modalBatch.modal('hide');
        });

		self.container.click(function() {
			if(self.is_bad) {
				setBad(false);
				showCanvas();
				return;
			}
		});

		$(document).keyup(function(e) {

			if(isModalOpen()) {
				return;
			}

	        switch(e.which) {
	        	case 65: // a
		        case 37: // left
		       	case 8: // backspace
		        	self.backBtn.click();
		        	break;
		        case 38: // up
		        	break;
		        case 32: // space
		        case 13: // enter
		        case 39: // right
		        case 68: // d
		        	self.nextBtn.click();
		        	break;
		        case 40: // down
		        	break;
		        case 82: // r
		        	break
		        case 67:
		        	self.clearBtn.click();
		        	break;
		        case 66:
		        	self.badBtn.click();
		        	break;
		        default: return; // exit this handler for other keys
		    }
		    e.preventDefault(); // prevent the default action (scroll / move caret)
		});


		self.batchBtn.click(function(e) {
			e.preventDefault();
			this.blur();

			if($(this).is('.disabled')) {
				return;
			}

			if(self.isSending) {
				return;
			}
			if(self.isNextLoading) {
				return;
			}

			openModalBatch(self.batch_id);
		});

		// next clicked
		self.nextBtn.click(function(e) {
			e.preventDefault();
			this.blur();

			if($(this).is('.disabled')) {
				return;
			}

			if(self.isSending) {
				return;
			}
			if(self.isNextLoading) {
				return;
			}

			if(self.next_id > 0) {
				sendAndRequest(null, null, self.next_id);
			} else {
				sendAndRequest(self.batch_id, self.sample_tag);
			}
		});

		// next other clicked
		self.nextOtherBtn.click(function(e) {
			e.preventDefault();
			this.blur();

			if($(this).is('.disabled')) {
				return;
			}

			if(self.isSending) {
				return;
			}
			if(self.isNextLoading) {
				return;
			}
			if(!self.has_other_tag) {
				return;
			}

			sendAndRequest(self.batch_id, null, null, self.sample_tag);
		});

		// back clicked
		self.backBtn.click(function(e) {
			e.preventDefault();
			this.blur();

			if($(this).is('.disabled')) {
				return;
			}

			if(self.isSending) {
				return;
			}
			if(self.isNextLoading) {
				return;
			}

			if(self.prev_id > 0) {
				sendAndRequest(null, null, self.prev_id);
			}
		});

		self.hardBtn.click(function(e) {
			e.preventDefault();
			this.blur();

			if($(this).is('.disabled')) {
				return;
			}

			$(this).toggleClass('checked');
			self.is_hard = $(this).is('.checked');
		});

		self.smallResolBtn.click(function(e) {
			e.preventDefault();
			this.blur();

			if($(this).is('.disabled')) {
				return;
			}

			$(this).toggleClass('checked');
			self.is_small_resol = $(this).is('.checked');
		});

		self.cytoBtn.click(function(e) {
			e.preventDefault();
			this.blur();

			if($(this).is('.disabled')) {
				return;
			}

			$(this).toggleClass('checked');
			self.is_cyto = $(this).is('.checked');
		});

		self.skipBtn.click(function(e) {
			e.preventDefault();
			this.blur();

			if($(this).is('.disabled')) {
				return;
			}

			$(this).toggleClass('checked-green');
			$(this).toggleClass('checked-blue');
			self.do_skip = $(this).is('.checked-blue');
			if(self.do_skip) {
				$(this).text("Режим просмотра");
			} else {
				$(this).text("Режим разметки");
			}
		});

		// bad clicked
		self.badBtn.click(function(e) {
			e.preventDefault();
			this.blur();

			if($(this).is('.disabled')) {
				return;
			}

			if(self.isSending) {
				return;
			}
			if(self.isNextLoading) {
				return;
			}

			if(self.is_bad) {
				setBad(false);
				return;
			}

			self.is_bad = true;
			if(self.do_skip) {
				setBad(true);
				return;
			}

			if(self.next_id > 0) {
				sendAndRequest(null, null, self.next_id);
			} else {
				sendAndRequest(self.batch_id, self.sample_tag);
			}
		});

		// clear clicked
		self.clearBtn.click(function(e) {
			e.preventDefault();
			this.blur();

			if($(this).is('.disabled')) {
				return;
			}

			if(self.isSending) {
				return;
			}
			if(self.isNextLoading) {
				return;
			}

			// reset values
			if(self.is_bad) {
				setBad(false);
				showCanvas();
			}
			if(self.is_cyto) {
				setCyto(false);
			}
			if(self.is_hard) {
				setHard(false);
			}
			if(self.is_small_resol) {
				setSmallResol(false);
			}

			Editor.clearData();
			Editor.draw();
		});

		// select type button clicked
		self.selectTypeBtns.click(function(e) {
			e.preventDefault();
			this.blur();

			if($(this).is('.disabled')) {
				return;
			}

			if(self.isSending) {
				return;
			}
			if(self.isNextLoading) {
				return;
			}

			if(self.is_bad) {
				setBad(false);
				showCanvas();
			}

			var nameId = $(this).attr('data-name');
			openModal(nameId);
		});

		$(window).bind('popstate', function(e) {
			// onPageReloaded(e.originalEvent.state)
		});

		// handle onresize only after resizing is finished
        $(window).bind('resizeEnd', function() {
		    //do , window hasn't changed size in 200ms
		    var viewW = self.container.width();
			var viewH = self.container.height();
			Editor.resize(viewW, viewH);
			Editor.draw();
		});
		$(window).resize(function() {
	        if(this.resizeTO) clearTimeout(this.resizeTO);
	        this.resizeTO = setTimeout(function() {
	            $(this).trigger('resizeEnd');
	        }, 10);
	    });
	}

	function isModalOpen() {
		return self.modal.hasClass('in');
	}

	function showBad() {
		self.canvasEl.css({"opacity": 0.0});
		self.imgMain.hide();
		self.loadingText.hide();
		self.emptyText.hide();
		self.badText.show();
		Editor.disable();
	}

	function showEmpty() {
		self.canvasEl.css({"opacity": 0.0});
		self.imgMain.hide();
		self.loadingText.hide();
		self.badText.hide();
		self.emptyText.show();
		Editor.disable();
	}

	function showLoading() {
		self.emptyText.hide();
		self.canvasEl.css({"opacity": 0.0});
		self.imgMain.hide();
		self.badText.hide();
		self.loadingText.show();
		Editor.disable();
	}

	function showCanvas() {
		self.loadingText.hide();
		self.emptyText.hide();
		self.badText.hide();
		self.imgMain.show();
        self.canvasEl.css({"opacity": 1.0});
        Editor.enable();
	}

	function onPageReloaded(state) {

		var sample_id;
		sample_id = state.sample_id;
		self.minPageIdx = state.min_page;
		self.pageIdx = state.page;

		if(sample_id == null) {
			// for empty state
			// replace state
			sendAndRequest(self.batch_id, self.sample_tag, null);
		} else {
			sendAndRequest(null, null, sample_id);
		}
	}

	function onDataLoaded(data) {

		// update location url and image src
		var imSrc = 'image/' + data.image_id + '/' + IMAGE_SZ;
		var locationUrl = '/marking?sample_id=' + data.id;
		self.remarking = data.cell_type_id < 0;
		if(self.remarking) {
			locationUrl += '&mode=remarking';
		}

		var state = {sample_id: data.id};
		window.history.replaceState(state, null, locationUrl);


		var historyData = {
			'prev_id': data.prev_id,
			'next_id': data.next_id
		};

		updateHistory(historyData);

		// update sidebar
		self.nextBtn.removeClass('disabled');
		self.badBtn.removeClass('disabled');
		self.cytoBtn.removeClass('disabled');
		self.smallResolBtn.removeClass('disabled');
		self.hardBtn.removeClass('disabled');
		self.clearBtn.removeClass('disabled');
		self.selectTypeBtns.removeClass('disabled');
		self.countBatchText.text('(' + data.batch_not_marked_count.toString() + ')');
		self.countText.text('Осталось всего: ' + data.not_marked_count.toString());

		if(data.next_id > 0 || !data.has_other_tag) {
			self.nextOtherBtn.addClass('disabled');
		} else {
			self.nextOtherBtn.removeClass('disabled');
		}

		// update current state
		self.has_other_tag = data.has_other_tag;
		self.imgMain.attr('src', imSrc);
		self.cell_type_id = data.cell_type_id;
		self.sample_id = data.id;
		self.is_empty = false;
		self.sampleTagLabel.text('Tэг: ' + data.sample_tag);
		self.sample_tag = data.sample_tag;

		if(!self.batch_id || self.batch_id != data.batch_id) {
			self.batch_id = data.batch_id;
			setLabelToBatchBtn(self.batch_id);
		}

		setHard(data.is_hard);
		setSmallResol(data.is_small_resol);
		setCyto(data.is_cyto);

		// update editor state
		Editor.setData(data.size.width, data.size.height, data.bboxes, data.detections, self.cell_type_id);
		Editor.unselectObjects();
		Editor.draw();

        setBad(data.is_bad);
        showCanvas();
	}

	function onEmptyData(data) {
		var imSrc = '';
		var locationUrl = '/marking';

		var historyData = {
			'prev_id': data.prev_id,
			'next_id': data.next_id
		};
		updateHistory(historyData);

		var state = {sample_id: null};
		window.history.replaceState(state, null, locationUrl);

		// update current state
		self.has_other_tag = false;
		self.imgMain.attr('src', imSrc);
		self.cell_type_id = null;
		self.sample_id = null;
		self.is_bad = false;
		self.is_empty = true;

		Editor.clearData();
		Editor.draw();

		self.nextBtn.addClass('disabled');
		self.nextOtherBtn.addClass('disabled');
		self.badBtn.addClass('disabled');
		self.clearBtn.addClass('disabled');
		self.cytoBtn.addClass('disabled');
		self.smallResolBtn.addClass('disabled');
		self.hardBtn.addClass('disabled');
		self.selectTypeBtns.addClass('disabled');
		self.sampleTagLabel.text('Нет изображений');
		self.countBatchText.text('(0)');
		self.countText.text('Осталось всего: ' + data.not_marked_count.toString());

		if(!self.batch_id || self.batch_id != data.batch_id) {
			self.batch_id = data.batch_id;
			setLabelToBatchBtn(self.batch_id);
		}

		showEmpty();
	}

	function requestPageData(batch_id=null, sample_tag=null, sample_id=null, exclude_tag=null) {
		// set loading state
		self.isNextLoading = true;

		// form url for request, known sample_id or known sample_tag
		var request_url = '/get_data';
		if(sample_id) {
			request_url += '?sample_id=' + sample_id;
		} else if (batch_id && sample_tag) {
			request_url += '?tag=' + sample_tag + '&batch_id=' + batch_id.toString();
		} else if (batch_id && exclude_tag) {
			request_url += '?batch_id=' + batch_id.toString() + '&exclude_tag=' + exclude_tag;
		} else if (batch_id) {
			request_url += '?batch_id=' + batch_id.toString();
		}

		var failed = function(errorMsg) {};

		var callParams = {
            url: request_url,
            type: 'GET',
            timeout: 10000,
            success: function(data, textStatus, jqXHR) {

                if(data && !data.empty) {
                    onDataLoaded(data);
                } else if(data) {
                    // show empty message, disable buttons
                    onEmptyData(data);
                } else if(failed != null) {
                    failed(jqXHR.responseText);
                }
            },
            error: function(jqXHR, textStatus, errorThrown) {
                if(failed != null) {
                	failed(jqXHR.responseText);
                }
            },
            complete: function() {
            	self.isNextLoading = false;
            },
            dataType: "json"
        };

		$.ajax(callParams);
	}

	function getUrlParameter(sParam) {
	    var sPageURL = decodeURIComponent(window.location.search.substring(1)),
	        sURLVariables = sPageURL.split('&'),
	        sParameterName,
	        i;

	    for (i = 0; i < sURLVariables.length; i++) {
	        sParameterName = sURLVariables[i].split('=');

	        if (sParameterName[0] === sParam) {
	            return sParameterName[1] === undefined ? true : sParameterName[1];
	        }
	    }
	}

	function sendCurrentData(callback, callbackFailed = null) {

		self.isSending = true;

		var dataSend;
		if(!self.is_bad) {
			var bboxes = Editor.getBoxes();
			var detections = Editor.getDetections();
			dataSend = {
				'sample_id': self.sample_id,
				'is_bad': false,
			 	'cell_type_id': self.cell_type_id,
			 	'bboxes': bboxes,
			 	'detections': detections,
			 	'is_hard': self.is_hard,
			 	'is_cyto': self.is_cyto,
			 	'is_small_resol': self.is_small_resol
			};
		} else {
			dataSend = {
				'sample_id': self.sample_id,
				'is_bad': true
			};
		}

		$.ajax({
			url: '/update_data',
			type: "post",
			dataType: 'json',
			data: JSON.stringify(dataSend),
    		contentType: "application/json",
    		success: function(data) {
    			if(data && data.status && data.status == 'OK') {
                    callback();
                } else if(callbackFailed) {
                    callbackFailed();
                }
    		},
    		failed: function(data) {
    			if(callbackFailed) {
    				callbackFailed();
    			}
    		},
			complete: function(data) {
				self.isSending = false;
			}
		});
	}

	function sendAndRequest(batch_id=null, sample_tag=null, sample_id=null, exclude_tag=null) {

		showLoading();

		if(!self.is_empty) {
			if(self.do_skip) {
				requestPageData(batch_id, sample_tag, sample_id, exclude_tag);
			} else {
				var callbackSend = function() {
					requestPageData(batch_id, sample_tag, sample_id, exclude_tag);
				}
				var callbackFailed = function() {
					showCanvas();
				}
				sendCurrentData(callbackSend, callbackFailed);
			}
		} else {
			requestPageData(batch_id, sample_tag, sample_id, exclude_tag);
		}
	}
};