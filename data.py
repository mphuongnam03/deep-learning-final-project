import os

# Đường dẫn tới các thư mục dữ liệu
train_dir = r'd:\Workspace\ai4life\dataset\train'
val_dir = r'd:\Workspace\ai4life\dataset\val'
test_dir = r'd:\Workspace\ai4life\dataset\test'

def count_images(folder):
    return len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

train_count = count_images(train_dir)
val_count = count_images(val_dir)
test_count = count_images(test_dir)
total = train_count + val_count + test_count

print(f'Train: {train_count} ({train_count/total:.2%})')
print(f'Val:   {val_count} ({val_count/total:.2%})')
print(f'Test:  {test_count} ({test_count/total:.2%})')
print(f'Total: {total}')