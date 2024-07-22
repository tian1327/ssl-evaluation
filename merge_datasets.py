# Define the file paths here
file1_path = 'data/semi_aves/T2T500+T2I0.25.txt'
file2_path = 'data/semi_aves/l_train.txt'
output_file_path = 'data/semi_aves/l_train_T2T500+T2I0.25.txt'

with open(file1_path, 'r') as file1:
    file1_lines = file1.readlines()

processed_file1_lines = []
for line in file1_lines:
    parts = line.strip().rsplit(' ', 2)  
    new_path = f"/retrieved/semi-aves/{parts[0]}"
    new_line = f"{new_path} {parts[1]}\n"  
    processed_file1_lines.append(new_line)


with open(file2_path, 'r') as file2:
    file2_lines = file2.readlines()


processed_file2_lines = []
for line in file2_lines:
    parts = line.strip().rsplit(' ', 1)  
    new_path = f"/dataset/semi-aves/{parts[0]}"
    new_line = f"{new_path} {parts[1]}\n"
    processed_file2_lines.append(new_line)


merged_lines = processed_file1_lines + processed_file2_lines

with open(output_file_path, 'w') as output_file:
    output_file.writelines(merged_lines)

print(f"Merging complete. The combined dataset has been written to {output_file_path}")
