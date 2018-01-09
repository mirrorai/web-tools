import json
import codecs

def main():

	with open('labels.json') as fp:
		labels = json.load(fp)

	with open('labels_eng.json') as fp:
		labels_eng = json.load(fp)

	brands_data = {}
	for label in labels:
		num_class = labels[label]
		if num_class not in brands_data:
			brands_data[num_class] = {}
		brands_data[num_class]['label'] = label

	for label in labels_eng:
		num_class = labels_eng[label]
		if num_class not in brands_data:
			brands_data[num_class] = {}
		brands_data[num_class]['name'] = label

	banks = [2,25,26,31,36,39,40,41,43,44,45,46,47,49,51,55,56,57,58,59,60,61,62,63,64]
	autos = [3,4,5,6,9,10,11,15,16,17,18,19,20,21,22,23,24,32,37,38,42,48,50,52,53,54]
	operators = [7,8,13,14]

	brands_list = []
	for num_class in brands_data:
		b = brands_data[num_class]
		obj = {}
		obj['class'] = num_class
		obj['name'] = b['name']
		obj['label'] = b['label']

		category = 'other'
		if num_class in banks:
			category = 'bank'
		elif num_class in autos:
			category = 'auto'
		elif num_class in operators:
			category = 'operator'

		obj['category'] = category
		brands_list.append(obj)

	cmd = {}
	cmd['name'] = 'add_brands'
	cmd['brands'] = brands_list

	cmds = {}
	cmds['commands'] = []
	cmds['commands'].append(cmd)

	with codecs.open('db_populate_brands.json', 'w', 'utf-8') as fp:
		json.dump(cmds, fp, indent=4, ensure_ascii=False)

if __name__ == '__main__':

	main()