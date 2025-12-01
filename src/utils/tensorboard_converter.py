import csv
from tensorboardX import SummaryWriter

csv_file = 'results.csv'
log_dir = 'runs/experiment_1' 

writer = SummaryWriter(log_dir)

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        epoch = int(row['epoch'])
        # Log các chỉ số bạn muốn lên TensorBoard
        writer.add_scalar('train/box_loss', float(row['train/box_loss']), epoch)
        writer.add_scalar('train/cls_loss', float(row['train/cls_loss']), epoch)
        writer.add_scalar('train/dfl_loss', float(row['train/dfl_loss']), epoch)
        writer.add_scalar('val/box_loss', float(row['val/box_loss']), epoch)
        writer.add_scalar('val/cls_loss', float(row['val/cls_loss']), epoch)
        writer.add_scalar('val/dfl_loss', float(row['val/dfl_loss']), epoch)
        writer.add_scalar('metrics/precision(B)', float(row['metrics/precision(B)']), epoch)
        writer.add_scalar('metrics/recall(B)', float(row['metrics/recall(B)']), epoch)
        writer.add_scalar('metrics/mAP50(B)', float(row['metrics/mAP50(B)']), epoch)
        writer.add_scalar('metrics/mAP50-95(B)', float(row['metrics/mAP50-95(B)']), epoch)
        writer.add_scalar('lr/pg0', float(row['lr/pg0']), epoch)
        writer.add_scalar('lr/pg1', float(row['lr/pg1']), epoch)
        writer.add_scalar('lr/pg2', float(row['lr/pg2']), epoch)

writer.close()
print(f"TensorBoard log saved to {log_dir}")