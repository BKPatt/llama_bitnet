import os

def combine_python_files(output_file):
    # Get the directory where the script is located
    directory = os.path.dirname(os.path.abspath(__file__))
    
    # Change the working directory to the script's directory
    os.chdir(directory)
    
    # Open the output file in write mode
    with open(output_file, 'w') as outfile:
        # Iterate through all the files in the directory
        for filename in os.listdir('.'):
            # Check if the file is a Python file and not the script itself
            if os.path.isfile(filename) and filename.endswith('.py') and filename != os.path.basename(__file__):
                # Write a separator to indicate the start of a new file
                outfile.write(f'# {filename}\n')
                outfile.write('#' + '='*78 + '\n')
                
                # Open the Python file and read its contents
                with open(filename, 'r') as infile:
                    # Write the contents to the output file
                    outfile.write(infile.read())
                    # Add two newlines for separation
                    outfile.write('\n\n')

    print(f'All Python files in {os.getcwd()} have been combined into {output_file}')

# Example usage
combine_python_files('combined_output.txt')
