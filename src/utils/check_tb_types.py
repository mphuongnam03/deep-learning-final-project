import pandas as pd

df = pd.read_csv('d:/Workspace/ai4life/tbx11k-simplified/data.csv')

# Kiểm tra các ảnh có TB
tb_df = df[df['target'] == 'tb']
print('=== PHÂN TÍCH ẢNH TB ===')
print(f'Tổng số ảnh TB: {len(tb_df)}')
print()

# Phân bố tb_type
print('Phân bố tb_type:')
print(tb_df['tb_type'].value_counts())
print()

# Kiểm tra bbox có khác nhau giữa active và latent không
print('=== MẪU BBOX THEO TB_TYPE ===')
for tb_type in ['active_tb', 'latent_tb']:
    samples = tb_df[tb_df['tb_type'] == tb_type].head(3)
    print(f'\n{tb_type.upper()}:')
    for _, row in samples.iterrows():
        bbox_str = str(row['bbox'])[:100]
        print(f"  {row['fname']}: bbox = {bbox_str}...")

# Kiểm tra cấu trúc bbox
print()
print('=== CẤU TRÚC BBOX ===')
sample_active = tb_df[tb_df['tb_type'] == 'active_tb'].iloc[0]['bbox']
sample_latent = tb_df[tb_df['tb_type'] == 'latent_tb'].iloc[0]['bbox']
print(f'Mẫu bbox active_tb: {str(sample_active)[:200]}')
print()
print(f'Mẫu bbox latent_tb: {str(sample_latent)[:200]}')
