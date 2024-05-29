import pandas as pd
import numpy as np
import plotly.express as px

# Generating synthetic data (if not already done)
np.random.seed(42)  # For reproducibility
n_samples = 500
dates_before = pd.date_range(start='2024-05-01', end='2024-05-15', periods=n_samples//2)
dates_after = pd.date_range(start='2024-06-01', end='2024-06-15', periods=n_samples//2)
dates = np.concatenate([dates_before, dates_after])

data = {
    'RQSTDTTM': dates,
    'BEGINDTTM': dates + pd.to_timedelta(np.random.randint(1, 5, n_samples), unit='m'),
    'ENDDTTM': dates + pd.to_timedelta(np.random.randint(5, 15, n_samples), unit='m'),
    'SECONDS': np.random.randint(30, 300, n_samples),
    'MINUTES': np.random.randint(1, 5, n_samples),
    'PRCSNAME': np.random.choice([f'JOB_{i}' for i in range(1, 167)], n_samples),
    'RUNSTATUSDESC': np.random.choice(['Success', 'Error'], n_samples, p=[0.9, 0.1]),
    'PRCSINSTANCE': np.random.randint(100000, 999999, n_samples),
    'OPRID': np.random.choice(['mcurtis', 'jdoe', 'asmith', 'bjohnson', 'ewilliams'], n_samples),
    'RUNCNTLID': np.random.choice(['EXTERNAL_PAYMENTS', 'INTERNAL_PAYMENTS'], n_samples)
}

df = pd.DataFrame(data)

# Calculate durations and request to begin time
df['DURATION'] = (df['ENDDTTM'] - df['BEGINDTTM']).dt.total_seconds() / 60  # Convert to minutes
df['UPGRADE'] = df['RQSTDTTM'] >= pd.Timestamp('2024-06-01')

# Calculate average pre and post-upgrade times for each job
pre_upgrade = df[df['UPGRADE'] == False].groupby('PRCSNAME')['DURATION'].mean().reset_index()
post_upgrade = df[df['UPGRADE'] == True].groupby('PRCSNAME')['DURATION'].mean().reset_index()

# Merge pre and post-upgrade data
comparison = pd.merge(pre_upgrade, post_upgrade, on='PRCSNAME', suffixes=('_Pre', '_Post'))
comparison['Difference (minutes)'] = comparison['DURATION_Post'] - comparison['DURATION_Pre']
comparison['Improvement (%)'] = (comparison['Difference (minutes)'] / comparison['DURATION_Pre']) * -100

# Add a column for coloring the scatter plot
comparison['Change'] = comparison['Difference (minutes)'].apply(lambda x: 'Improvement' if x < 0 else 'Regression' if x > 0 else 'No Change')

# Create scatter plot
fig = px.scatter(comparison, x='DURATION_Pre', y='DURATION_Post', color='Change',
                 title='Job Completion Times Pre vs. Post Upgrade',
                 labels={'DURATION_Pre': 'Pre-Upgrade Time (minutes)', 'DURATION_Post': 'Post-Upgrade Time (minutes)'},
                 hover_data=['PRCSNAME', 'Difference (minutes)', 'Improvement (%)'])

# Add line of equality (y=x) to show no change
fig.add_shape(
    type='line',
    line=dict(dash='dash'),
    x0=0, y0=0,
    x1=comparison['DURATION_Pre'].max(),
    y1=comparison['DURATION_Pre'].max()
)

# Show plot
fig.show()

########
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Generating synthetic data (if not already done)
np.random.seed(42)  # For reproducibility
n_samples = 500
dates_before = pd.date_range(start='2024-05-01', end='2024-05-15', periods=n_samples//2)
dates_after = pd.date_range(start='2024-06-01', end='2024-06-15', periods=n_samples//2)
dates = np.concatenate([dates_before, dates_after])

data = {
    'RQSTDTTM': dates,
    'BEGINDTTM': dates + pd.to_timedelta(np.random.randint(1, 5, n_samples), unit='m'),
    'ENDDTTM': dates + pd.to_timedelta(np.random.randint(5, 15, n_samples), unit='m'),
    'SECONDS': np.random.randint(30, 300, n_samples),
    'MINUTES': np.random.randint(1, 5, n_samples),
    'PRCSNAME': np.random.choice([f'JOB_{i}' for i in range(1, 167)], n_samples),
    'RUNSTATUSDESC': np.random.choice(['Success', 'Error'], n_samples, p=[0.9, 0.1]),
    'PRCSINSTANCE': np.random.randint(100000, 999999, n_samples),
    'OPRID': np.random.choice(['mcurtis', 'jdoe', 'asmith', 'bjohnson', 'ewilliams'], n_samples),
    'RUNCNTLID': np.random.choice(['EXTERNAL_PAYMENTS', 'INTERNAL_PAYMENTS'], n_samples)
}

df = pd.DataFrame(data)

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ... (your existing code for data generation remains the same) ...

# Calculate durations
df['DURATION'] = (df['ENDDTTM'] - df['BEGINDTTM']).dt.total_seconds() / 60  # Convert to minutes
df['UPGRADE'] = df['RQSTDTTM'] >= pd.Timestamp('2024-06-01')

# Calculate statistics pre and post upgrade (no grouping to get overall stats)
stats_pre = df[df['UPGRADE'] == False]['DURATION'].agg(['min', 'mean', 'median', 'max', 'std'])
stats_post = df[df['UPGRADE'] == True]['DURATION'].agg(['min', 'mean', 'median', 'max', 'std'])

# Create a summary DataFrame for the overall statistics
summary_data = {
    'Category': ['Pre-Upgrade', 'Post-Upgrade'],
    'Min': [stats_pre['min'], stats_post['min']],
    'Mean': [stats_pre['mean'], stats_post['mean']],
    'Median': [stats_pre['median'], stats_post['median']],
    'Max': [stats_pre['max'], stats_post['max']],
    'Std': [stats_pre['std'], stats_post['std']]
}

df_summary = pd.DataFrame(summary_data)

# Calculate statistics per job
stats_pre_per_job = df[df['UPGRADE'] == False].groupby('PRCSNAME')['DURATION'].agg(['min', 'mean', 'median', 'max', 'std']).reset_index()
stats_post_per_job = df[df['UPGRADE'] == True].groupby('PRCSNAME')['DURATION'].agg(['min', 'mean', 'median', 'max', 'std']).reset_index()

# Merge pre and post upgrade stats
stats_pre_per_job.columns = ['PRCSNAME', 'Pre-Upgrade Min', 'Pre-Upgrade Mean', 'Pre-Upgrade Median', 'Pre-Upgrade Max', 'Pre-Upgrade Std']
stats_post_per_job.columns = ['PRCSNAME', 'Post-Upgrade Min', 'Post-Upgrade Mean', 'Post-Upgrade Median', 'Post-Upgrade Max', 'Post-Upgrade Std']

comparison = pd.merge(stats_pre_per_job, stats_post_per_job, on='PRCSNAME')

# Calculate the difference in means
comparison['Mean Difference'] = comparison['Post-Upgrade Mean'] - comparison['Pre-Upgrade Mean']
comparison['Relative Difference (%)'] = (comparison['Mean Difference'] / comparison['Pre-Upgrade Mean']) * 100

# Sort by the absolute difference (descending) to prioritize both increases and decreases
comparison['Abs Mean Difference'] = comparison['Mean Difference'].abs()
comparison.sort_values(by='Abs Mean Difference', ascending=False, inplace=True)


# Create traces for the overall summary
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_summary['Mean'],
    y=df_summary['Category'],
    mode='markers',
    error_x=dict(type='data', array=df_summary['Std']),
    name='Overall',
    marker=dict(color='black', size=12),
    hovertemplate="%{y}:<br>" +
                  "Mean: %{x:.2f} mins<br>" +
                  "Min/Max: %{customdata[0]:.2f} / %{customdata[1]:.2f} mins<br>" +
                  "Std: %{customdata[2]:.2f} mins",
    customdata=df_summary[['Min', 'Max', 'Std']]
))

# Create traces for individual jobs (same as before, but for all jobs)
fig.add_trace(go.Scatter(
    x=comparison['Pre-Upgrade Mean'],
    y=comparison['PRCSNAME'],
    mode='markers',
    error_x=dict(
        type='data',
        symmetric=False,
        array=comparison['Pre-Upgrade Max'] - comparison['Pre-Upgrade Mean'],
        arrayminus=comparison['Pre-Upgrade Mean'] - comparison['Pre-Upgrade Min']
    ),
    name='Pre-Upgrade',
    marker=dict(color='blue'),
    hovertemplate=(
        "Job: %{y}<br>" +
        "Pre-Upgrade Mean: %{x:.2f} mins<br>" +
        "Pre-Upgrade Min/Max: %{customdata[0]:.2f} / %{customdata[1]:.2f} mins<br>" +
        "Pre-Upgrade Std: %{customdata[2]:.2f} mins"
    ),
    customdata=comparison[['Pre-Upgrade Min', 'Pre-Upgrade Max', 'Pre-Upgrade Std']]
))


# Post-Upgrade "Candlesticks" (for all jobs)
fig.add_trace(go.Scatter(
    x=comparison['Post-Upgrade Mean'],
    y=comparison['PRCSNAME'],
    mode='markers',
    error_x=dict(
        type='data',
        symmetric=False,
        array=comparison['Post-Upgrade Max'] - comparison['Post-Upgrade Mean'],
        arrayminus=comparison['Post-Upgrade Mean'] - comparison['Post-Upgrade Min']
    ),
    name='Post-Upgrade',
    marker=dict(color='green'),
    hovertemplate=(
        "Job: %{y}<br>" +
        "Post-Upgrade Mean: %{x:.2f} mins<br>" +
        "Post-Upgrade Min/Max: %{customdata[0]:.2f} / %{customdata[1]:.2f} mins<br>" +
        "Post-Upgrade Std: %{customdata[2]:.2f} mins<br>" +
        "Mean Difference: %{customdata[3]:+.2f} mins<br>" +  # Display change in mean
        "Relative Difference: %{customdata[4]:+.2f}%"       # Display percentage change
    ),
    customdata=comparison[['Post-Upgrade Min', 'Post-Upgrade Max', 'Post-Upgrade Std', 'Mean Difference', 'Relative Difference (%)']]
))

# Figure layout
fig.update_layout(
    title='Job Duration Statistics: Pre-Upgrade vs. Post-Upgrade (All Jobs)',
    xaxis_title='Duration (Minutes)',
    yaxis_title='Job Name',
    height=3000,  
    width=1200,
    yaxis={'categoryorder': 'array', 'categoryarray': comparison['PRCSNAME']}
)

fig.show()

