from torchref.restraints_helper import read_link_definitions
    

link_dict, link_list = read_link_definitions()

print(link_dict['TRANS']['bonds'])
    
