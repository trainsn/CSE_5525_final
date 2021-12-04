
def convert2NeoData(database):
    # table is a dict
    graph_data = []
    for key in database.keys():
        table_names = database[key]['table_names']
        columns_names = database[key]['column_names']
        edges  = []
        nodes = [{
            'type':'node',
            'id': 'core'+key,
            'labels':['db_name'],
            'properties':{
                'level': 0,
                'name': 'DataBase',
                'primary':0
            }
        }]
        for ele in table_names:
            nodes.append({
                'type':'node',
                'id': ele,            
                'labels': ['table_name'],
                'properties':{
                    'level':1,
                    'name': ele,
                    'primary':0
                }
            })
            edges.append({
                'id': ele+"_"+key,
                'type': 'relationship',
                'startNode': 'core'+key,
                'endNode': ele,
                'label': 'has_table_',
                'properties':{}
            })
        for ele in columns_names:
            if ele[0]>-1:
                if ele[0] in database[key]['primary_keys']:
                    temp=1
                else: 
                    temp=0
                nodes.append({
                    'type':'node',
                    'id': str(ele[0])+ele[1],
                    'labels': ['column_name'],
                    'properties':{
                        'level':2,
                        'name': ele[1],
                        'primary':temp
                    }
                })
                edges.append({
                    'id': ele[1]+"_"+table_names[ele[0]],
                    'type': 'relationship',
                    'startNode': table_names[ele[0]],
                    'endNode': str(ele[0])+ele[1],
                    'label': 'has_column',
                    'properties':{
                    }
                })
        foreign_keys = database[key]['foreign_keys']
        for pair in foreign_keys:
            s = columns_names[pair[0]]
            t = columns_names[pair[1]]
            edges.append({
                'id': 'f_'+str(pair[0])+str(pair[1]),
                'type':'relationship',
                'startNode': str(s[0])+s[1],
                'endNode': str(t[0])+ t[1],
                'label':'is_foreign',
                'properties':{
                    
                }
            })

        one_graph = {
            'name': key,
            'neodata': {
                'results':[{
                    'columns':[],
                    'data':[{
                        'graph':{
                            'nodes': nodes,
                            'relationships': edges
                        }
                    }]
                }],
                'errors':[]
            }
        }
        graph_data.append(one_graph)
    return graph_data