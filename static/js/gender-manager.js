
var GenderManager =  new function() {
	var self = this;

	// entry point
	self.init = function(is_male) {

        self.is_male = is_male;
	    self.isSending = false;
	    self.isNextLoading = false;

		initElements();
		initHandlers();
	}

	// init UI variables
	function initElements() {
	}

	function updateCardBackgroundColor(elem) {

	    var changed = elem.find('.btn-gender').is('.active');
	    var is_bad = elem.find('.btn-gender-bad').is('.active');
	    var is_hard = elem.find('.btn-gender-hard').is('.active');

	    elem.removeClass('bg-info bg-warning bg-danger');

	    if(is_hard || is_bad) {
            elem.addClass('bg-danger');
	    } else {
	        var is_male_new = (self.is_male && !changed) || (!self.is_male && changed);
	        if(is_male_new) {
	            elem.addClass('bg-info');
	        } else {
	            elem.addClass('bg-warning');
	        }
	    }
	}

	// init handlers
	function initHandlers() {

		$(document).keyup(function(e) {

	        switch(e.which) {
	        	case 65: // a
		        case 37: // left
		       	case 8: // backspace
		        	break;
		        case 38: // up
		        	break;
		        case 32: // space
		        case 13: // enter
		        case 39: // right
		        case 68: // d
		        	break;
		        case 40: // down
		        	break;
		        case 82: // r
		        	break
		        case 67:
		        	break;
		        case 66:
		        	break;
		        default: return; // exit this handler for other keys
		    }
		    e.preventDefault(); // prevent the default action (scroll / move caret)
		});

		// next clicked
		$("img.img-card").click(function(e) {
			e.preventDefault();
			this.blur();

			if($(this).is('.disabled')) {
				return;
			}

            $(this).closest('.sample-card').find('button.btn-gender').click();
		});

		$('button.btn-gender').click(function(e) {
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

			$(this).toggleClass('active');
			var changed = $(this).is('.active');

			var parent = $(this).closest('.sample-card');
			updateCardBackgroundColor(parent);

            var elem = parent.find('.gender-data-container').first();
			elem.attr('data-is-changed', changed ? "1" : "0");
		});

		$('button.btn-gender-hard').click(function(e) {
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

			$(this).toggleClass('active');
			var is_hard = $(this).is('.active');

			var parent = $(this).closest('.sample-card');
			updateCardBackgroundColor(parent);

            var elem = parent.find('.gender-data-container').first();
			elem.attr('data-is-hard', is_hard ? "1" : "0");
		});

		// bad clicked
		$('button.btn-gender-bad').click(function(e) {
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

			$(this).toggleClass('active');
			var is_bad = $(this).is('.active');

			var parent = $(this).closest('.sample-card');
			updateCardBackgroundColor(parent);

            var elem = parent.find('.gender-data-container').first();
			elem.attr('data-is-bad', is_bad ? "1" : "0");
		});

		// bad clicked
		$('#form_send button.btn').click(function(e) {
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

			$(this).toggleClass('active');
            prepareSubmitData();
			$(this).closest('#form_send').submit();
		});
	}

	function getDataJson() {
	    var data = {};
	    $('.sample-card').each(function(i, obj) {
	        var obj = $(obj);
            var data_container = obj.find('.gender-data-container').first();
            var sample_id = data_container.data('sample-id');

            data_item = {}

            if (typeof sample_id === 'undefined') {
                return;
            }
            var is_changed = data_container.data('is-changed');
            if (!(typeof is_changed === 'undefined') && is_changed === 1) {
                data_item['is_changed'] = 1;
            } else {
                data_item['is_changed'] = 0;
            }
            var is_hard = data_container.data('is-hard');
            if (!(typeof is_hard === 'undefined') && is_hard === 1) {
                data_item['is_hard'] = 1;
            }
            var is_bad = data_container.data('is-bad');
            if (!(typeof is_bad === 'undefined') && is_bad === 1) {
                data_item['is_bad'] = 1;
            }
            data[sample_id] = data_item;
        });

	    return JSON.stringify(data);
	}

	function prepareSubmitData() {
        var form = $('#form_send');
        form.find('input#is_male').val(self.is_male);
        form.find('input#gender_data').val(getDataJson());
	}
};