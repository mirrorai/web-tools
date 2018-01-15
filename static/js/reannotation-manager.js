
var ReannotationManager =  new function() {
	var self = this;

	// entry point
	self.init = function() {

	    self.isSending = false;
	    self.updatingTasks = {};

		initElements();
		initHandlers();
	}

	// init UI variables
	function initElements() {
	}

	// init handlers
	function initHandlers() {

		$('button.btn-train').click(function(e) {
			e.preventDefault();
			this.blur();

			if($(this).is('.disabled')) {
				return;
			}

            var btn = $(this)
            btn.toggleClass('disabled');

            var state = btn.data('state');
            if (typeof state !== 'undefined' && state == 1) {
                stopTrain(btn, function(success) {
                    btn.toggleClass('disabled');
                });
            } else {
                triggerTrain(btn, function(success) {
                    btn.toggleClass('disabled');
                });
            }
		});
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
                    btn.text('Train');
                    btn.data('state', 0);
			        btn.removeClass('btn-warning').addClass('btn-success');
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

        var container = btn.closest('.list-group-item');
        container.find('.progress_nanobar').remove();

        div = $('<div class="progress_nanobar lead"><div></div><div>0%</div><div>...</div><div>&nbsp;</div></div>');
        container.append(div);

        // create a progress bar
        var nanobar = new Nanobar({
            bg: '#44f',
            target: div[0].childNodes[0]
        });

		$.ajax({
			url: '/trigger_train',
			type: "post",
			dataType: 'json',
			data: JSON.stringify(dataSend),
    		contentType: "application/json",
    		success: function(data, status, request) {
    		    if(data.status && data.status=='ok') {
                    status_url = data.status_url;

                    btn.data('task-id', data.task_id);
                    btn.data('stop-url', data.stop_url);
                    btn.data('status-url', data.status_url);
                    btn.data('state', 1);

                    btn.text('Stop');
			        btn.removeClass('btn-success').addClass('btn-warning');
                    self.updatingTasks[data.task_id] = true;
                    updateProgress(data.task_id, status_url, btn, nanobar, div[0]);
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
                if ('result' in data) {
                    // show result
                    $(status_div.childNodes[3]).text('Result: ' + data['result']);
                    btn.text('Train');
                    btn.data('state', 0)
                    btn.removeClass('btn-warning').addClass('btn-success');
                    btn.removeClass('disabled');
                }
                else {
                    // something unexpected happened
                    $(status_div.childNodes[3]).text('Result: ' + data['state']);
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