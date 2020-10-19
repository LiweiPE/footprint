# import rhinoscriptsyntax as rs
import json
import matplotlib.pyplot as plt

#prompt the user for a file to import
# filter = "JSON file (*.json)|*.json|All Files (*.*)|*.*||"
# filename = 'rmse_3.56.json'
filename = 'epoch_500_batchsize_16.json'
#Read JSON data into the datastore variable
if filename:
    with open(filename, 'rb') as f:
        datastore = json.load(f)

# print (datastore['val_loss'][99])
# print (datastore['val_loss'][299])
# print (datastore['val_loss'][499])

# summarize history for loss
plt.plot(datastore['loss'])
plt.plot(datastore['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid(True)
plt.ylim((0,0.05))
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#
# #Read JSON data into the datastore variable
# if filename:
#     with open(filename, 'r') as f:
#         datastore = json.load(f)