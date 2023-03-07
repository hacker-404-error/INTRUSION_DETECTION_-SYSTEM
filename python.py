import os

# specify the path where you want to create the folder
path = "C:\\Users\\Prita\\Desktop\\CURRENT AFFAIRS\\CURRENT AFFAIRS\\04. APR - 2023"

# specify the name of the folder you want to create

for n in range(10,32):
    folder_name = "{}. {} - APR - 23".format(n,n)
    # create the folder if it doesn't exist already
    if not os.path.exists(os.path.join(path, folder_name)):
        os.mkdir(os.path.join(path, folder_name))
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")