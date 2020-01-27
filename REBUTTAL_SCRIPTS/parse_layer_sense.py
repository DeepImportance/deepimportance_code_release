import ast

fr = open('layer_sens.log', 'r')
lines = fr.readlines()

interested_model = 'LeNet4'
interested_layer = 6

orig_tot = 0
rel_tot = 0
rel_lines = []
for line in lines:
    line = line.strip()
    res = ast.literal_eval(line)
    if res['model_name'] == interested_model and res['layer'] == interested_layer and res['qgran'] == 3:
        orig_tot += res['orig_coverage']
        rel_tot += res['rel_coverage']

print(orig_tot)
print(rel_tot)
