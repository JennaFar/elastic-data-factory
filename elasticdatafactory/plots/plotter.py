# -*- coding: utf-8 -*-
# importing necessary libraries to data processing
# import functions for interacting with the operating system
import os
# import sys functions and variables to manipulate Python runtime environment
import sys
# import AWS SDK for Python
import boto3
# import awswrangler package
import awswrangler as wr
# import sagemaker
import sagemaker
# import pandas for relational data analysis and manipulation
import pandas as pd
# import numpy to create and manipulate arrays
import numpy as np
# import module for working with files and directories
from pathlib import Path
# import matplotlib for plotting
import matplotlib.pyplot as plt
# import patches from matplotlib
from matplotlib.patches import Patch
# import line collection for multicolor line segments
from matplotlib.collections import LineCollection
# import sns for plotting graphs
import seaborn as sns
# import Graph Objects from plotly for funnel visualization
from plotly import graph_objects as go
# import colors from plotly for funnel visualization
from plotly import colors
# metrics to evaluate model performance
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score,
)
# import logger for logs
import logging

# load logger
logger = logging.getLogger('')


def plot_histogram(x: list, y: list, xlabel: str, ylabel: str, hist_type='bivariate', nrows=1, ncols=1):
    """
    This function creates a plot of a histogram displaying bin edges on the x-axis
    and the corresponding frequencies on the y-axis for multiple features.

    Parameters
    ----------
    x : list
        A list containing input data for x variable
    y : list
        A list containing input data for y variable
    xlabel : str
        A string describing x variable
    ylabel : str
        A string describing y variable
    hist_type : str
        Either 'univariate' or 'bivariate'. If bivariate, nrows and ncols will be default to one, showing only one sub-plot.
        If univariate, one subgraph will display one variable
    nrows : int, default 1
        An integer representing number of rows for subplots
    ncols : int, default 1
        An integer representing number of columns for subplots

    Returns
    -------
    None

    Examples
    --------
    # Plot chart
    >>>x = [1,2,3,4,5,6,4,3,2,4,5,3,4,2,4,5,6,4,3,4,2,5,1,5,6,7]
    >>>y = [9,8,7,8,6,7,8,7,6,5,6,7,8,6,7,8,9,0,8,7,6,7,8,8,7,6]
    >>>xlabel = 'xlabel'
    >>>ylabel = 'ylabel'
    >>>plot_histogram(x=x, y=y, xlabel=xlabel, ylabel=ylabel, hist_type='bivariate')
    >>>plot_histogram(x=x, y=y, xlabel=xlabel, ylabel=ylabel, hist_type='univariate')
    """

    # create subplots
    if hist_type not in ['bivariate', 'univariate']:
        logger.error(f'[ERROR] provided his_type <{hist_type}> is invalid, please select from bivariate or univariate!')
        return
    if hist_type == 'bivariate':
        fig, ax = plt.subplots(nrows=1,
                               ncols=1,
                               figsize=(8, 4))
        ax.hist([x, y],
                bins='auto',
                range=(min(x + y), max(x + y)),
                stacked=False,
                density=True,
                label=(str(xlabel), str(ylabel)))

        ax.legend()
        ax.set_title('Distribution of $X$ and $Y$ Observations')
        ax.yaxis.tick_right()
        plt.show()
    else:
        fig, ax = plt.subplots(nrows=1,
                               ncols=2,
                               figsize=(8, 4))

        ax[0].hist([x],
                   bins='auto',
                   range=(min(x), max(x)),
                   stacked=False,
                   density=True,
                   label=(str(xlabel)))
        ax[0].legend()
        ax[0].set_title('Distribution of $X$ Observations')
        ax[0].yaxis.tick_right()

        ax[1].hist([y],
                   bins='auto',
                   range=(min(y), max(y)),
                   stacked=False,
                   density=True,
                   label=(str(ylabel)))
        ax[1].legend()
        ax[1].set_title('Distribution of $Y$ Observations')
        ax[1].yaxis.tick_right()
        plt.show()
        
# plot univariate distribution of covariate
def plot_bar(data, var_labels, x_var, y_var=None, width=8, height=5):
    
    """
    This function creates a bar plot variable displaying binned values on the x-axis
    and the corresponding frequencies on the y-axis.

    Parameters
    ----------
    data : pd.dataFrame
        data stored in dataFrame for visualization
    var_labels : str
        A string representing label for X-axis
    x_var : str
        A string describing X variable to plot
    y_var : str
        A string describing Y variable, By default, it is assigned None
    width : int
        width of the resulting plot, by default, it is 8
    height : int
        height of the resulting plot, by default, it is 5

    Returns
    -------
    None

    Examples
    --------
    # Generate Bar Plot
    >>> plot_bar(data=df_loanidentifier_lead_visit_event_milestone, 
                var_labels='Marketing Channels', 
                x_var='first_marketingchannel_finder')
    
    """
    
    
    palette_size = len(data[x_var].unique())
    sequential_colors = sns.color_palette("RdPu", palette_size)
    plt.figure(figsize=(8, 5)) # this creates a figure 8 inch wide, 5 inch high
    plt.title(f'Distribution of {var_labels}', weight='bold').set_fontsize('14')
    if y_var is None:
        x = data.groupby(x_var).size().reset_index(name='count')[x_var].values
        y = data.groupby(x_var).size().reset_index(name='count')['count'].values
        order = data.groupby(x_var).size().reset_index(name='count').sort_values('count')[x_var].values
        ax = sns.barplot(data, x=x, y=y, order=order, palette=sequential_colors)
    if y_var is not None:
        ax = sns.barplot(data, x=x_var, y=y_var, palette=sequential_colors)
    for bar in ax.containers:
        ax.bar_label(bar, size=10, label_type='edge')
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)
    ax.set_xlabel(var_labels, fontsize=12, fontdict=dict(weight='bold'))
    ax.set_ylabel('Count', fontsize=12, fontdict=dict(weight='bold'))
    plt.tight_layout()
    plt.show()
    

# plot univariate distribution of covariate
def plot_histogram_kde(data, var_labels, x_var, y_var=None, width=8, height=5, x_tick_rotation=45, kde=False):
    
    """
    This function creates a hitogram plot of a a given variable displaying binned values on the x-axis
    and the corresponding frequencies on the y-axis.

    Parameters
    ----------
    data : pd.dataFrame
        data stored in dataFrame for visualization
    var_labels : str
        A string representing label for X-axis
    x_var : str
        A string describing X variable to plot
    y_var : str
        A string describing Y variable, By default, it is assigned None
    width : int
        width of the resulting plot, by default, it is 8
    height : int
        height of the resulting plot, by default, it is 5
    x_tick_rotation : int
        Rotate X-axis label by certain value, by default, it is set to 90
    kde : str
        Plot univariate or bivariate distributions using kernel density estimation
        By default, it is assigned False

    Returns
    -------
    None

    Examples
    --------
    # Plot chart
    >>> plot_histogram_kde(data=df,
                           var_labels='Online Credit Events By Time', 
                           x_var='online_credit_estdatetime',
                           width=8,
                           height=5,
                           x_tick_rotation=45,
                           kde=False)
    
    """
    
    # check if X variable is date/time
    if data[x_var].dtypes == 'datetime64[ns]':
        data.sort_values(by=[x_var], inplace=True)
    plt.figure(figsize=(width, height)) # this creates a figure 8 inch wide, 5 inch high
    plt.title(f'Distribution of {var_labels}', weight='bold').set_fontsize('14')
    if y_var is None:
        ax = sns.histplot(data, x=x_var, alpha=.4, kde=kde, color="purple")
    if y_var is not None:
        ax = sns.histplot(data, x=x_var, y=y_var, alpha=.4, kde=kde, color="purple")
    #ax.bar_label(ax.containers[0], fmt='%.2f%%') # label on top of each bar
    plt.setp(ax.get_xticklabels(), rotation=x_tick_rotation, horizontalalignment='right', fontsize=10)
    ax.set_xlabel(var_labels, fontsize=12, fontdict=dict(weight='bold'))
    ax.set_ylabel('Count', fontsize=12, fontdict=dict(weight='bold'))
    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % 2 == 0:  # every other label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.tight_layout()
    plt.show()
    
def heatmap(data, symmetric=True, width=8, height=5, x_tick_rotation=90, linewidths=.2, vmin=-1, vmax=1, cmap='PiYG', annot=True):

    """
    This function creates a matrix plot or a rectangular data plot as a color-encoded matrix.

    Parameters
    ----------
    data : pd.dataFrame
        2D dataset that can be coerced into an ndarray. If a Pandas DataFrame is provided, 
        the index/column information will be used to label the columns and rows
    symmetric : boolean
        If True, data will not be shown in cells above the diagonal because of the 
        nature of the matrix, i.e., Rij = Rji,
    width : int
        width of the resulting plot, by default, it is 8
    height : int
        height of the resulting plot, by default, it is 5
    x_tick_rotation : int
        Rotate X-axis label by certain value, by default, it is set to 90
    linewidths : float
        Width of the lines that will divide each cell.
    vmin : float
        Minimum values to anchor the colormap. By default, it is assigned -1
    vmax : float
        Minimum values to anchor the colormap. By default, it is assigned +1
    cmap : matplotlib colormap name or object
        The mapping from data values to color space. If not provided, 
        the default colormap will be 'PiYG'
    annot : boolean
        If True, write the data value in each cell.

    Returns
    -------
    None

    Examples
    --------
    # Plot chart
    >>> sns.heatmap(correlation_matrix, 
                    symmetric=True, 
                    linewidths=.2, 
                    vmin=-1, 
                    vmax=1, 
                    cmap='PiYG', 
                    annot=True)
    
    """

    if symmetric is True:
        mask = np.zeros_like(correlation_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
    else:
        mask = None
    plt.figure(figsize=(width, height))
    sns.heatmap(data, mask=mask, linewidths=linewidths, vmin=vmin, vmax=vmax, cmap=cmap, annot=annot)
    plt.xticks(rotation=x_tick_rotation)
    plt.show()
    
# function to plot conversion funnel
def plot_funnel(dataFrame, funnel_type='', funnel_type_col='', landing_page='', landing_page_col='',
                touchpoints=[], width=1200, height=600, title='Conversion Funnel', label_metric='value+percent initial'):
    
    """
    Plot funnel to show fallout rate b/w each step or touchpoints 
    
    Parameters
    ----------
    dataFrame : pd.dataFrame
        Data stored in dataFrame
    funnel_type : str
        Type of funnel if data represents more than one. By default, it is set to an empty string
    funnel_type_col : str
        Field name representing various types of funnel if data represents more than one. By default, it is set to an empty string
    landing_page : str
        Name of landing page if data represents more than one. By default, it is set to an empty string
    landing_page_col : str
        Field name representing various landing pages if data represents more than one. By default, it is set to an empty string
    touchpoints : list
        List of touchpoints or milestones in a conversion funnel. It must match the labels assigned to each column in dataFrame
    width : int
        A scalar value to assign plot width
    height : int
        A scalar value to assign plot height
    title : str
        A string representing title of the plot
    label_metric : str
        A string represting label metric that is displayed on the funnel. 
        Valid values are 'value+percent initial', 'value+percent previous', 'value+percent total' 
        By default, it is set to 'value+percent initial' if no value is provided

    Returns
    -------
    None
    
    Examples
    --------
    
    # set touchpoints
    >>> touchpoints = ['first_marketingchannel_finder', 'first_loanpurpose', 'first_homedescription', 'first_propertyuse', 
                       'first_timeframetopurchase', 'first_firsttimebuyer', 'first_hasrealestateagent', 'first_creditrating']
    # plot conversion funnel
    >>> plot_funnel(dataFrame=df,  
                    funnel_type='Purchase',
                    funnel_type_col='first_loanpurpose',
                    landing_page='Ql Lander', 
                    landing_page_col='first_landing_sitesection', 
                    touchpoints=touchpoints,
                    width=1000,
                    height=600,
                    title='Conversion Funnel',
                    label_metric='value+percent total'
                   )
    
    """
    
    logger.info('[INFO] finding number of visitors at each touchpoint...')
    filter_criteria = ''
    try:
        if dataFrame[(dataFrame[funnel_type_col] == funnel_type) & (dataFrame[landing_page_col] == landing_page)].shape[0] > 0:
            logger.info('[INFO] funnel type & landing page detected...')
            filter_criteria = 'funnel and landing'
        elif dataFrame[dataFrame[funnel_type_col] == funnel_type].shape[0] > 0:
            logger.info('[INFO] funnel type detected...')
            filter_criteria = 'funnel'
        elif dataFrame[dataFrame[landing_page_col] == landing_page].shape[0] > 0:
            logger.info('[INFO] landing page detected...')
            filter_criteria = 'landing'
    except:
        pass

    if filter_criteria == 'funnel and landing':
        dataFrame =  dataFrame[(dataFrame[funnel_type_col] == funnel_type) & (dataFrame[landing_page_col] == landing_page)]
    elif filter_criteria == 'funnel':
        dataFrame = dataFrame[dataFrame[funnel_type_col] == funnel_type]
        logger.warning('[WARNING] landing page not detected, skipping landing page specific filters...')
    elif filter_criteria == 'landing':
        dataFrame = dataFrame[dataFrame[landing_page_col] == landing_page]
        logger.warning('[WARNING] funnel type not detected, skipping funnel type specific filters...')
            
    # count visitors through touchpoints
    if touchpoints:
        dataFrame = dataFrame[touchpoints]
    check_entries = dataFrame.progress_apply(lambda x: x.count(), axis=0)
    # create a dataframe through this visitor count
    count_visitors = pd.DataFrame(data=list(zip(dataFrame.columns.tolist(), check_entries)), 
                                  columns=['Touchpoint', 'Visitors']
                                 )
    # set plot color gradient
    color_map = colors.n_colors('rgb(0, 255, 0)', 'rgb(255, 0, 0)', len(touchpoints)+1, colortype='rgb')
    # plot funnel
    fig = go.Figure(go.Funnel(
        y = count_visitors[count_visitors['Touchpoint'].isin(touchpoints)]['Touchpoint'].to_list(),
        x = count_visitors[count_visitors['Touchpoint'].isin(touchpoints)]['Visitors'].to_list(),
        textposition = 'outside',
        textinfo = label_metric,
        opacity = 0.75, 
        marker = {'color': color_map[:-1], 'line': {"width": [3] * len(touchpoints)}},
        connector = {'line': {'color': 'royalblue', 'dash': 'dot', 'width': 3}})
                   )
    # update layout
    fig.update_layout(
        title='<b>' + title + ' ' + f'({funnel_type})' + '</b>',
        extendfunnelareacolors=True,
        autosize=False,
        width=width,
        height=height,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    # display funnel
    fig.show()
    
def plot_cv_indices(cv, X, y, group, n_splits=5, line_width=10):
    """
    Create a sample plot for indices of a cross-validation object.
    
    Parameters
    ----------
    cv : cross validation method
        Cross-validator such as TimeSeriesSplit(), StratifiedKFold(), GroupKFold()
    X : list
        A list containing input data for X variable
    y : list
        A list containing input data for y variable
    group : list
        A list containing group association for each observation
    n_splits : int
        An integer representing number of splits to perform on data
    line_width : int
        An integer representing line width for partition bars
        
    Returns
    -------
    None
    
    Examples
    --------
    # Plot partitions
        plot_cv_indices(cv=tscv, X=X, y=y, group=groups, n_splits=n_splits)
    
    """
    
    # assign color map to groups and classes
    cmap_data = plt.cm.Paired
    # assign color map to train and test split
    cmap_cv = plt.cm.coolwarm
    # Create a figure 8 inch wide, 5 inch high
    fig, ax = plt.subplots(figsize=(8, 5))
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0
        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=line_width,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )
    # Plot the data classes
    ax.scatter(range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=line_width, cmap=cmap_data)
    # Plot the data groups at the end
    ax.scatter(range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=line_width, cmap=cmap_data)
    # Formatting
    yticklabels = list(range(1, n_splits+1)) + ['class', 'group']
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample Index",
        ylabel="CV Iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, len(X)],
    )
    # create plot legend to display
    ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ['Testing set', 'Training set'],
        loc=(1.02, 0.8),
    )
    # assign title to the plot
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    # allow legent overlap
    plt.tight_layout()
    # display plot
    plt.show()
    
def PlotConfusionMatrix(y_test, y_pred, threshold=0.5):
    
    """
    This function computes confusion matrix to evaluate the accuracy of a classification.
        
    Parameters
    ----------
    y_test : list
        A list containing true binary labels or binary label indicators 
    y_pred : list
        A list containing target scores, or probability estimates of the positive class
    threshold : float, default 0.5
        A floating point representing threshold for classifying binary labels
        
    Returns
    -------
    tn : int
        Number of True Negatives
    fp : int
        Number of False Positives
    fn : int
        Number of False Negatives
    tp : int
        Number of True Positives
        
    Examples
    --------
    # Plot matrix
        PlotConfusionMatrix(y_test, y_pred, threshold)
    
    """
    
    # threshold all raw predicted values
    y_pred = [1 if x >= threshold else 0 for x in y_pred]
    # plot confusion matrix using scikit
    cfn_matrix = confusion_matrix(y_test, y_pred)
    # unravel confusion matrix to retrieve values
    tn, fp, fn, tp = cfn_matrix.ravel()
    # creat a plot
    fig, ax = plt.subplots(figsize=(8, 5))
    # create a color map
    sns.heatmap(cfn_matrix,cmap='coolwarm_r', linewidths=0.5, annot=True, ax=ax, fmt='g')
    # assign title to this plot
    plt.title('Confusion Matrix')
    # assign Y-label
    plt.ylabel('Real Classes')
    # assign X-label
    plt.xlabel('Predicted Classes')
    # display plot
    plt.show()
    # display Classification Report 
    print('')
    print('---Classification Report---')
    print(classification_report(y_test, y_pred))
    
    return tn, fp, fn, tp, fig
        
def plot_roc_pr(y_test, y_pred, positive_label=1, thresholds_every=10, camp='jet'):
    
    """
    This function creates a plot ROC curve with corresponding Area Under the Curve (AUC) metric in  
    addition to Precision-Recall curve with Average Precision (AP) metric. AP summarizes a 
    precision-recall curve as the weighted mean of precisions achieved at each threshold, 
    with the increase in recall from the previous threshold used as the weight
        
    Parameters
    ----------
    y_test : list
        A list containing true binary labels or binary label indicators 
    y_pred : list
        A list containing target scores, or probability estimates of the positive class
    positive_label : int
        An integer representing label of the positive class. Only applied to binary y_true
    thresholds_every : int
        An integer representing the interval between threshold values shown on plots
    camp : str
        A string describing color map. By default, this is set to 'jet'

    
    Returns
    -------
    None
        
    Examples
    --------
    # Plot chart
        plot_roc_pr(y_test, y_pred, positive_label=1, thresholds_every=200)
    
    """
    
    # create subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=False)
    # generate ROC curve
    # fp: false positive rates. tp: true positive rates
    fp, tp, thresholds = roc_curve(y_test, y_pred, pos_label=positive_label)
    roc_auc = auc(fp, tp)
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([fp, tp]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # cutoff threshold when the last value is above 1
    thresholds = [x-1 if x > 1.0 else x for x in thresholds]
    # Create a continuous norm to map from data points to colors
    lc = LineCollection(segments, cmap=camp)
    # Set the values used for colormapping
    lc.set_array(thresholds)
    lc.set_linewidth(2)
    line = ax[0].add_collection(lc)
    # plot colorbar on the right
    fig.colorbar(line, ax=ax[0])
    # plot line to indicate lower bound for model performance
    ax[0].plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=1)
    ax[0].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax[0].set_xlabel('False positives rate', fontsize=12, fontweight='bold')
    ax[0].set_ylabel('True positives rate', fontsize=12, fontweight='bold')
    ax[0].set_xlim(-0.01, 1.01)
    ax[0].set_ylim(-0.01, 1.01)
    ax[0].legend([lc], ['ROC Curve (area = %0.2f)' % roc_auc], loc="lower right")
    colorMap = plt.get_cmap(camp, len(thresholds)).reversed()
    for i in range(0, len(thresholds), thresholds_every):
        threshold_value_with_max_four_decimals = str(thresholds[i])[:5]
        ax[0].text(fp[i] + 0.06, tp[i] + 0.01, threshold_value_with_max_four_decimals, fontdict={'size': 8}, color=colorMap(i/len(thresholds)));
    
    # generate PR curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred, pos_label=positive_label)
    # compute avergae precision
    ap = average_precision_score(y_test, y_pred)
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([recall, precision]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # cutoff threshold when the last value is above 1
    thresholds = [x-1 if x > 1.0 else x for x in thresholds]
    # Create a continuous norm to map from data points to colors
    lc = LineCollection(segments, cmap=camp)
    # Set the values used for colormapping
    lc.set_array(thresholds)
    lc.set_linewidth(2)
    line = ax[1].add_collection(lc)
    # plot colorbar on the right
    fig.colorbar(line, ax=ax[1])
    # plot line to indicate lower bound for model performance
    random_performance = np.round(len(y_test[y_test == 1]) / len(y_test), 3)
    ax[1].plot([0, 1], [random_performance, random_performance], color='navy', linestyle='--', linewidth=1)
    ax[1].set_title('Precision-Recall (PR) Curve', fontsize=14, fontweight='bold')
    ax[1].set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax[1].set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax[1].set_xlim(-0.01, 1.01)
    ax[1].set_ylim(-0.01, 1.01)
    ax[1].legend([lc], ['Precision-Recall Curve (AP = %0.2f)' % ap], loc="lower right")
    colorMap = plt.get_cmap(camp, len(thresholds))
    for i in range(0, len(thresholds), thresholds_every+300):
        threshold_value_with_max_four_decimals = str(thresholds[i])[:5]
        ax[1].text(recall[i] - 0.07, precision[i] - 0.05, threshold_value_with_max_four_decimals, fontdict={'size': 8}, color=colorMap(i/len(thresholds)));
    # plot subplots
    plt.show()
    
    return roc_auc, ap, fig

class PlotTuningProgress:
    
    """Exports raw user data to an Amazon Web Services (AWS) S3 bucket with CSV uploads.
    
    Attributes
    ----------
    data_frame : Pandas DataFrame
        A DataFrame containing data to be imported or exported as comma-separated values
    tuning_job_name : str
        A string containing the name of Amazon Web Services (AWS) S3 bucket
    sagemaker_client : boto3.client('sagemaker')
        A low-level client representing Amazon SageMaker Service, and 
        provides APIs for creating and managing SageMaker resources
    
    """
    
    def __init__(self, df, tuning_job_name, sagemaker_client):
        # report all results from hyperparameter tuning job
        self.df = df
        self.tuner = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name)
        self.tooltips = []
        for key in self.tuner.tuning_ranges.keys():
            self.tooltips.append(key)
        tuning_job_result = sagemaker_client.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=tuning_job_name
        )
        self.metric_name = tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['MetricName']

    def hover_tooltip(self):
        
        """save dataframe to a .csv file in Amazon Web Services (AWS) S3 bucket
        
        Parameters
        ----------
        self : object
            ImportExportCSV class instance.
        index : boolean, default=False
            A data object representing row names (index).
    
        Returns
        -------
        status : boolean
            A data object representing status of export.
        """
        
        # create Hovertemplate for plotly
        hovertemp = "<b>Timestamp: </b> %{x}"
        hovertemp += "<b>Objective Value: </b> %{y}"
        for index, key in enumerate(self.tuner.tuning_ranges.keys()):
            hovertemp += f"<b>{key}: </b>" + "%{customdata[" + f"{index}" + "]} <br>"
        hovertemp += "<extra></extra>"
        
        return hovertemp
    
    def plot_params(self):
        
        """save dataframe to a .csv file in Amazon Web Services (AWS) S3 bucket
        
        Parameters
        ----------
        self : object
            ImportExportCSV class instance.
        index : boolean, default=False
            A data object representing row names (index).
    
        Returns
        -------
        status : boolean
            A data object representing status of export.
        """
        
        # create figure
        fig = go.Figure(layout=dict(width=800, height=500, hovermode='closest', showlegend=True))
        # create a scatter plot
        trace = go.Scatter (x=self.df['TrainingStartTime'],
                            y=self.df['FinalObjectiveValue'],
                            mode='markers',
                            name='<b>Tuning Jobs</b>',
                            customdata=self.df[self.tooltips],
                            hovertemplate=self.hover_tooltip(),
                            marker=dict(size=10, color='Aquamarine', line=dict(width=2, color='DarkSlateGrey'))
                            )
        # add scatter plot to figure
        fig.add_trace(trace)
        # update layout
        fig.update_layout(legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01
            ),
        margin=dict(l=20, r=20, t=50, b=20),
        title='<b>Hyperparameter Tuning Results</b>',
        title_x=0.5,
        )
        # update axes
        fig.update_xaxes(tickprefix="<b>",
                         ticksuffix ="</b><br>",
                         title_text= "<b> Timestamp </b>" )
        fig.update_yaxes(tickprefix="<b>",
                         ticksuffix ="</b><br>",
                         title_text= f"<b> Objective Metric ({self.metric_name}) </b>", 
                         tickformat='.3f')
        fig.show()