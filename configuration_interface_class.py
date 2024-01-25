
import sys
class QueueingNetwork_Applicaition:
    def __init__(self):
        self.active_cap = None
        self.deactive_t = None
        self.total_layers = []
        self.adjacent_list = {}
        self.buffer_size_list = []
        self.lambda_list = []
        self.miu_list = []
       
    # Application Functions 
    def application_closing_handler(self,layer):
        if str(layer)  == 'q': 
            print("Thank you for using the Queuing Network Builder. \nHave a nice day!")
            sys.exit()
        

    def input_validation(self,prompt):
        while True:
            user_input = input(prompt)
            try:
                return int(user_input)
            except ValueError:
                print("Please ensure that the input value is specified as an integer.")
            
    def confirm_prompt_validation(self,prompt):
        while True:
            user_input = input(prompt)
            if user_input.lower() in ['n', 'y']:
                return user_input.lower()
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        
    def layer_builder(self,layer_counter , servers_in_layer , server_counter):
        """
            This function builds a single layer of the queuing network. 
        """
        # Create an indexed list of servers in the current layer and append to the main listt
        server_list = list(range(server_counter, server_counter + servers_in_layer)) # server indices for the current layer 
        self.total_layers.append(server_list)
        server_counter += servers_in_layer
        if layer_counter == 0:
            return server_counter 
        if layer_counter > 0: 
            # the servers available for connection are those in the previous layer 
            servers_available_for_connection = self.total_layers[layer_counter-1]
        return server_list,servers_available_for_connection,server_counter

    def queue_builder(self,server_list,servers_available_for_connection):
        """
            This function builds the queue connecting the servers in the current layer to the servers in the previous layer.
        """
        internal_server_list = server_list[:]
        while internal_server_list:
            servers_available_for_connection_copy = servers_available_for_connection[:]
            prompt = f"The servers available for selection are {', '.join(str(server) for server in server_list)}"
            prompt += "\nPlease select the server you would like to configure: "
            selected_server = self.input_validation(prompt)
            if selected_server not in internal_server_list:
                print(f"Please ensure that the server you select is in the list of servers available for selection.")
                continue
            else:
                self.adjacent_list.setdefault(selected_server, [])
                done_configuring = False 
                internal_server_list.remove(selected_server)
                while not done_configuring:
                    
                    prompt = f"The servers available for connection are {', '.join(str(server) for server in servers_available_for_connection_copy)}"
                    prompt += "\nPlease select the server you would like to connect to: "
                    connecting_server = self.input_validation(prompt)   
                    if connecting_server not in servers_available_for_connection_copy:   
                        print(f"\nPlease ensure that the server you select is in the list of servers available for connection.")
                        continue
                    
                    else:
                        # Remove a server from the list of servers available for connection as long as the list still has servers in it
                        servers_available_for_connection_copy.remove(connecting_server)
                        buffer_size = self.input_validation(f"\nPlease enter the buffer size for the queue connecting server {connecting_server} to server {selected_server}: ")
                        lambda_value = self.input_validation(f"\nPlease enter the arrival rate for the queue connecting server {connecting_server} to server {selected_server}: ")
                        miu_value = self.input_validation(f"\nPlease enter the service rate for the queue connecting server {connecting_server} to server {selected_server}: ")
                        self.adjacent_list[selected_server].append(connecting_server)
                        self.buffer_size_list.append(buffer_size)
                        self.lambda_list.append(lambda_value)
                        self.miu_list.append(miu_value) 
                        if not servers_available_for_connection_copy:
                            # If there are no more servers available for connection, then the queue configuration for the current server is complete
                            print(f"\nYou have successfully configured all the possible connections for server {selected_server}.")
                            done_configuring = True
                        else:
                            ending_prompt = self.confirm_prompt_validation(f"\nWould you like to configure another queue for server {selected_server}? (y/n): ")
                            if ending_prompt == 'n':
                                print(f"\nYou have successfully configured the queue for server {selected_server}.")
                                done_configuring = True
                            else:
                                continue
        
                        
    def application_runtime(self):
        # Runtime Loop
        print("====== Welcome to the Queuing Network Builder =====")
        # Initializing data structures 

        layer_counter,server_counter  = (0,0)
        
        while True: 
            # Initializing loop structures
            current_server_list = []
            servers_available_for_connection = []
            
            # Building the queuing network
            if layer_counter == 0:
                self.active_cap = self.input_validation("Please enter the maximum active capacity of the servers across the network: ")
                self.deactive_t = self.input_validation("Please enter the maximum deactive capacity of the servers across the network: ")
            else:
                pass    
            servers_in_layer = self.input_validation(f"Building layer number {str(layer_counter+1)} of the queuing network. \nPlease enter the number of servers in this layer: ")
            if layer_counter == 0:
                # The first layer of the queuing network is the only layer that does not have any queues connecting it to the previous layer
                server_counter = self.layer_builder(layer_counter , servers_in_layer , server_counter)
            else:
                current_server_list,servers_available_for_connection,server_counter = self.layer_builder(layer_counter , servers_in_layer , server_counter)  
                self.queue_builder(current_server_list,servers_available_for_connection)
            layer_counter += 1
            closing_prompt = self.confirm_prompt_validation("Would you like to configure another layer? (y/n): ")
            if closing_prompt == 'n':
                print("You have successfully configured the queuing network.")
                print("\nThank you for using the Queuing Network Builder. \nHave a nice day!")
                print("The following are the variables for the queuing network you have configured:")
                print(f"Active Capacity: {self.active_cap}")
                print(f"Deactive Capacity: {self.deactive_t}")
                print(f"Lambda Values: {self.lambda_list}")
                print(f"Miu Values: {self.miu_list}")
                print(f"Buffer Sizes: {self.buffer_size_list}")
                print(f"Adjacent List: {self.adjacent_list}")
                sys.exit()
            else:
                continue
            
    def serve_variables(self):
        return (self.lambda_list , self.miu_list, self.active_cap , self.deactive_t , self.adjacent_list , self.buffer_size_list)

if __name__ == '__main__':
    app = QueueingNetwork_Applicaition()
    app.application_runtime()
    
        
