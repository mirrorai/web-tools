
var ReannotationManager =  new function() {
	var self = this;

	// entry point
	self.init = function(update_url) {

	    self.stopActions = false;
	    self.nanobars = {};
	    self.update_url = update_url;

		initElements();
		initHandlers();

		self.update_timer = setInterval(requestUpdate, 2000);
	}

	self.update = function(data) {
	    self.stopActions = true;
        for(var i = 0; i < data.length; i++) {
            var data_for_problem = data[i];
            var problem_name = data_for_problem['name_id'];
            var label = data_for_problem['name'];
            var enabled = data_for_problem['enabled'];
            var container = $('#' + problem_name + '_content');
            var items = container.find('#' + problem_name + '_items');
            var annotation = items.find('#' + problem_name + '_annotation');
            var metrics = items.find('#' + problem_name + '_metrics');
            var manager = items.find('#' + problem_name + '_manager');

            if(enabled) {
                container.removeClass('disabled');
                items.show();
                annotation.attr('href', data_for_problem['annotation_url']);
                if('stats' in data_for_problem) {
                    var stats = data_for_problem['stats'];
                    annotation.find('span.count-total').text(stats['total']);
                    annotation.find('span.count-to-check').text(stats['to_check']);
                    annotation.find('span.count-checked').text(stats['total_checked']);
                    annotation.find('span.count-reannotated').text(stats['total_reannotated']);
                }
                if('metrics' in data_for_problem) {
                    metrics.show();
                    var metrics_data = data_for_problem['metrics'];
                    metrics.find('span.accuracy').text(metrics_data['accuracy']);
                    metrics.find('span.accuracy-grouth').text(metrics_data['accuracy_grouth']);
                } else {
                    metrics.hide();
                }
                if('tasks' in data_for_problem) {
                    manager.show();
                    var task_data = data_for_problem['tasks'];
                    for(var t = 0; t < task_data.length; t++) {
                        var task = task_data[t];
                        var row = manager.find('#' + problem_name + '_' + task['problem_type']);
                        updateStatusInfo(problem_name, row, task);
                    }
                } else {
                    manager.hide();
                }
            } else {
                container.addClass('disabled');
                items.hide();
            }
            container.find('#' + problem_name + '_header').text(label);
        }
        self.stopActions = false;
	}

	function updateStatusInfo(problem_name, row, task) {
        var btn = row.find('button.btn');
        var status_container = row.find('.status-container');
        btn.attr('data-start-url', task['start_url']);
        btn.attr('data-stop-url', task['stop_url']);

        if(btn.attr('data-start-url') === '#') {
            btn.addClass('disabled');
        } else {
            btn.removeClass('disabled');
        }

        if(!task['is_finished'] && task['tasks'].length == 0) {
            // no info
            btn.removeClass('btn-warning').addClass('btn-success');
            btn.text(task['label']);
            btn.attr('data-state', 'off');
        } else if(!task['is_finished']) {
            // process
            btn.removeClass('btn-success').addClass('btn-warning');
            btn.text('Stop');
            btn.attr('data-state', 'on');
        } else {
            // finished
            btn.removeClass('btn-warning').addClass('btn-success');
            btn.text(task['label']);
            btn.attr('data-state', 'off');
        }

        for(var t = 0; t < task['tasks'].length; t++) {
            var task_data = task['tasks'][t];
            var k_fold  = task_data['k_fold'];
            var fold_id = k_fold === null ? '' : k_fold
            var nanobar_container = status_container.find('#' + k_fold + '.progress-nanobar-container');
            if(nanobar_container.length == 0) {
                // create
                var div_str = '<div id="' + k_fold + '"class="progress-nanobar-container">';
                div_str += '<div class="progress-nanobar"></div>';
                div_str += '<div class="progress-percent"></div>';
                div_str += '<div class="progress-status"></div>';
                div_str += '</div>';

                nanobar_container = $(div_str);
                status_container.append(nanobar_container);
            }
            nanobar_container.show();

            var progress_nanobar = nanobar_container.find('.progress-nanobar');
            var progress_percent = nanobar_container.find('.progress-percent');
            var progress_status = nanobar_container.find('.progress-status');

            if('finished_ts' in task_data && task_data['finished_ts'] === null) {
                // show process
                progress_nanobar.show();
                progress_percent.show();
                progress_status.show();

                var nanobar_id = '#' + problem_name + '_' + task['problem_type'] + '_' + fold_id;
                if(!(nanobar_id in self.nanobars)) {
                    var nanobar = new Nanobar({
                        bg: '#44f',
                        target: progress_nanobar[0],
                    });
                    self.nanobars[nanobar_id] = nanobar;
                }
                var nanobar = self.nanobars[nanobar_id];

                var percent = 100.0 * task_data['progress'];
                percent = percent.toFixed(2);
                nanobar.go(percent);
                progress_percent.text(percent + '%');
                progress_status.text(task_data['status']);

            } else if('finished_ts' in task_data) {
                // finished

                progress_nanobar.hide();
                progress_percent.hide();
                progress_status.show();

                progress_status.text(task_data['status'] + ', finished: ' + task_data['finished_ts']);
            } else {
                // not started any task yet

                progress_nanobar.hide();
                progress_percent.hide();
                progress_status.hide();
            }
        }
	}

	function requestUpdate() {
        $.getJSON(self.update_url, function(data) {
            self.update(data);
        });
	}

	// init UI variables
	function initElements() {
	}

	// init handlers
	function initHandlers() {

		$('button.btn').click(function(e) {
			e.preventDefault();
			this.blur();

			if($(this).is('.disabled')) {
				return;
			}

			if(self.stopActions) {
			    return;
			}

            var btn = $(this);
            btn.toggleClass('disabled');

            sendRequest(btn, function(success) {});
		});
	}

	function sendRequest(btn, callback) {
	    self.stopActions = true;

		var request_url = btn.attr('data-start-url');
		console.log(btn.attr('data-state'));
		if(btn.attr('data-state') === 'on') {
		    request_url = btn.attr('data-stop-url');
		}

		$.ajax({
			url: request_url,
			type: 'get',
    		success: function(data, status, request) {
    		    if(data.status && data.status=='ok') {
                    self.update(data.problems);
                    callback(true);
    		    } else {
    		        callback(false)
    		    }
    		},
    		error: function(data) {
    			callback(false);
    		},
			complete: function(data) {
				self.stopActions = false;
			}
		});
	}

	function createProgressbarForBtn(btn) {
	    var container = btn.closest('.list-group-item');
        container.find('.progress_nanobar').remove();

        var div = $('<div class="progress_nanobar lead"><div></div><div>0%</div><div>...</div></div>');
        container.append(div);

        // create a progress bar
        var nanobar = new Nanobar({
            bg: '#44f',
            target: div[0].childNodes[0]
        });

        return {'nanobar': nanobar, 'elem': div}
	}

	function stopTrain(btn, callback) {
	    self.isSending = true;

        var container = btn.closest('.list-group-item');

		$.ajax({
			url: btn.data('stop-url'),
			type: "post",
			dataType: 'json',
			data: JSON.stringify({}),
    		contentType: "application/json",
    		success: function(data, status, request) {
    		    if(data.status && data.status=='ok') {
                    toggleBtn(btn);
			        delete self.updatingTasks[btn.data('task-id')];
			        container.find('.progress_nanobar').remove();
                    callback(true);
    		    } else {
    		        callback(false)
    		    }

    		},
    		error: function(data) {
    			callback(false);
    		},
			complete: function(data) {
				self.isSending = false;
			}
		});
	}

	function triggerTrain(btn, callback) {
        self.isSending = true;

        var task = btn.data('task-name');
		var dataSend = { 'task': task };

		$.ajax({
			url: '/trigger_train/' + task,
			type: 'get',
    		success: function(data, status, request) {
    		    if(data.status && data.status=='ok') {
                    var status_url = data.status_url;
                    toggleBtn(btn, data);
                    var bar_data = createProgressbarForBtn(btn);
                    self.updatingTasks[data.task_id] = true;
                    updateProgress(data.task_id, status_url, btn, bar_data['nanobar'], bar_data['elem'][0]);
                    callback(true);
    		    } else {
    		        callback(false)
    		    }

    		},
    		error: function(data) {
    			callback(false);
    		},
			complete: function(data) {
				self.isSending = false;
			}
		});
	}

	function toggleBtn(btn, data=null) {
	    if(data === null) {
	        btn.text('Train');
            btn.data('state', 'off')
            btn.removeClass('btn-warning').addClass('btn-success');
	    } else {
            btn.data('task-id', data.task_id);
            btn.data('stop-url', data.stop_url);
            btn.data('status-url', data.status_url);
            btn.data('state', 'on');

            btn.text('Stop');
            btn.removeClass('btn-success').addClass('btn-warning');
	    }
	}

	function updateProgress(task_id, status_url, btn, nanobar, status_div) {

	    if(!(task_id in self.updatingTasks)) {
	        return;
	    }

        // send GET request to status URL
        $.getJSON(status_url, function(data) {

            if(!(task_id in self.updatingTasks)) {
                return;
            }

            // update UI
            percent = data['current'] * 100 / data['total'];
            percent = percent.toFixed(2);
            nanobar.go(percent);
            $(status_div.childNodes[1]).text(percent + '%');
            $(status_div.childNodes[2]).text(data['status']);
            if (data['state'] != 'PENDING' && data['state'] != 'PROGRESS') {
                if (data['state'] == 'SUCCESS') {
                    // show result
                    // $(status_div.childNodes[3]).text('Result: success');
                    toggleBtn(btn);
                }
                else {
                    // something unexpected happened
                    console.log(data);
                    $(status_div.childNodes[3]).text('Error during training occured.');
                    toggleBtn(btn);
                }
            }
            else {
                // rerun in 2 seconds
                setTimeout(function() {
                    updateProgress(task_id, status_url, btn, nanobar, status_div);
                }, 2000);
            }
        });
    }
};