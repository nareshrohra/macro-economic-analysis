# import relevant libraries
from datetime import datetime, date
import math
from xml.dom.expatbuilder import TEXT_NODE
import pandas as pd, numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from numpy import cov
import ipywidgets as widgets
from IPython.core.display import display, Javascript
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

index_list = None
events_list = None
average_trading_days_in_year = 200

#String Constants
TEXT_SELECT = 'Select'
TEXT_CHART_TYPE_CANDLESTICK = 'Candlestick'
TEXT_CHART_TYPE_GROWTH = 'Growth'
TEXT_CHART_TYPE_GROWTH_PCT = 'Growth (%)'

class DataScrapper:
    def merge_scrapped_data(self, folder, last_file_index):
        path = ".\\Scrapped Data\\" + folder + "\\"
        merged_data = pd.read_csv(path + "data.csv")
        
        for i in range(1, last_file_index + 1):
            filename = "data (" + str(i) + ")" + ".csv"
            csv_data = pd.read_csv(path + filename)
            merged_data = merged_data.append(csv_data)
            
        merged_data.to_csv(".\\Scrapped Data\\" + folder + "\\full-data.csv")

class DataProcessorBase:
    __index_name = None
    __from_date = None
    __to_date = None
    for_month = None
    for_day = None

    def set_date_range_filter(self, from_date, to_date):
        self.__from_date = pd.Timestamp(from_date)
        self.__to_date = pd.Timestamp(to_date)

    def set_index_name(self, index_name):
        self.__index_name = index_name

    def set_for_month_filter(self, for_month):
        self.for_month = for_month

    def set_for_day_filter(self, for_day):
        self.for_day = for_day

    def add_growth_cols(self, df, open_price_col, close_price_col, prefix = None):
        start_value = df.iloc[0][open_price_col]
        if prefix is not None:
            df[prefix + '_Growth'] = df[close_price_col] - start_value
            df[prefix + '_Growth_Pct'] = df[prefix + '_Growth'] / start_value
        else:
            df['Growth'] = df[close_price_col] - start_value
            df['Growth_Pct'] = df['Growth'] / start_value
    
    def get_yearly_data(self, df, colors):
        yearly_df = pd.DataFrame(columns = ['Date', 'Year', 'High', 'Low', 'Open', 'Close', 'Turnover'])
        prev_close = 1
        for year in df['Year'].unique():
            current_year_df = df[df['Year'] == year]
            year_first_day = df[df['Date'] == (current_year_df['Date'].min())].iloc[0]
            year_last_day = df[df['Date'] == (current_year_df['Date'].max())].iloc[0]
            yearly_df = yearly_df.append({
                'Date': current_year_df['Date'].min(),
                'Year': year,
                'High': current_year_df['High'].max(),
                'Low': current_year_df['Low'].min(),
                'Open': year_first_day['Open'],
                'Close': year_last_day['Close'],
                'Growth': year_last_day['Close'] - prev_close,
                'Growth_Perc': min((year_last_day['Close'] - prev_close) / prev_close * 100, 100)
                #'Turnover': df[df['Year'] == year]['Turnover'].sum()
            }, ignore_index=True)
            prev_close = year_last_day['Close']
        yearly_df["GrowthColor"] = np.where(yearly_df["Growth_Perc"] < 0, colors[1], colors[0])
        return yearly_df

    # this is trading days span not calendar days
    def add_ath_cols(self, df, ath_min_gap_days, ath_max_gap_days):
        for price_type in ['Open', 'High', 'Low']:
            df.loc[(df[price_type] == '-') | (df[price_type] == 0), price_type] = pd.to_numeric(df[(df[price_type] == '-') | (df[price_type] == 0)]['Close'])

        for price_type in ['Open', 'High', 'Low']:
            df[price_type] = pd.to_numeric(df[price_type])

        df['Is ATH'] = False
        for row_index in range(0, len(df['High'])):
            row = df.iloc[row_index]
            df.at[row_index, 'ATH'] = df.iloc[0:row_index+1]['High'].max()
            searchStartIndex = max(row_index - ath_max_gap_days, 0)
            df.at[row_index, 'Is ATH'] = row['High'] == df.iloc[searchStartIndex: (row_index + 1 + ath_min_gap_days)]['High'].max()

        df['Is Lowest Low'] = False
        ath_indexes = df.index[df['Is ATH'] == True]
        for i in range(0, len(ath_indexes)):
            ath_index = ath_indexes[i]
            if i < len(ath_indexes) - 1:
                ath_next_index = ath_indexes[i + 1]
            else:
                ath_next_index = len(df['Date'])
            lowest_low_index = df.iloc[ath_index:ath_next_index]['Low'].idxmin()
            df.at[lowest_low_index, 'Is Lowest Low'] = True
        
        df['Is Lowest Low'] = df['Is Lowest Low'].fillna(False)

        df['Low from ATH'] = -(df['ATH'] - df['Low'])
        df['Perc low from ATH'] = (df['Low from ATH'] / df['ATH']) * 100

        self.add_cycles(df)

    def add_cycles(self, df):
        ath_row_nos = df.index[df['Is ATH'] == True]
        cycle_start_date = df.iloc[0]['Date']
        cycle_start_open_price = df.iloc[0]['Open']
        for index in range(0, len(ath_row_nos)):
            ath_row_no = ath_row_nos[index]
            lowest_low_row_no = df[ath_row_no:][df['Is Lowest Low'] == True].index[0]
            lowest_low_row = df.iloc[lowest_low_row_no]
            
            df.loc[df['Date'].between(cycle_start_date, lowest_low_row['Date']), 'Cycle'] = index + 1
            df.loc[df['Date'].between(cycle_start_date, lowest_low_row['Date']), 'Returns Within Cycle'] = df.loc[df['Date'].between(cycle_start_date, lowest_low_row['Date']), 'Close'] - cycle_start_open_price
            df.loc[df['Date'].between(cycle_start_date, lowest_low_row['Date']), 'Perc Returns Within Cycle'] = (df.loc[df['Date'].between(cycle_start_date, lowest_low_row['Date']), 'Close'] - cycle_start_open_price) / cycle_start_open_price * 100
            
            if lowest_low_row_no + 1 < len(df['Date']):
                cycle_start_date = df.iloc[lowest_low_row_no + 1]['Date']
                cycle_start_open_price = df.iloc[lowest_low_row_no + 1]['Open']

        df['Cycle'] = df['Cycle'].fillna(index + 1).astype(int)
    
    def add_date_cat_cols(self, df):
        df['Year'] = pd.DatetimeIndex(df['Date']).year
        df['Month'] = pd.DatetimeIndex(df['Date']).month_name()
        df['Quarter'] = pd.DatetimeIndex(df['Date']).quarter

    def get_data_from_sheet(self, sheet_name):
        #filepath = self.__filename
        filepath = "./IndexData/" + sheet_name + ".xlsx"
        df = pd.read_excel(filepath, sheet_name=sheet_name, index_col=0, engine='openpyxl').reset_index()
        return df

    def get_series_data_from_sheet(self, sheet_name):
        df = self.get_data_from_sheet(sheet_name=sheet_name)
        df = df[np.isnat(df['Date']) == False]
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        if self.__from_date is not None:
            df = df[df['Date'] >= self.__from_date]
            df = df[df['Date'] <= self.__to_date]
        return df

    def get_index_data(self):
        df = self.get_series_data_from_sheet(self.__index_name)
        return df
    
    def get_merged_data(self, sheet1_name, sheet2_name, cols_to_rename = None, how = 'inner'):
        df1 = self.get_series_data_from_sheet(sheet1_name)
        df2 = self.get_series_data_from_sheet(sheet2_name)

        merged_df = pd.merge(df1, df2, how=how, on=['Date'])
        if cols_to_rename is not None:
            merged_df = merged_df.rename(columns=cols_to_rename)
        return merged_df

class ListReader (DataProcessorBase):
    def read_lists(self):
        global index_list, events_list
        lists = self.get_data_from_sheet('Lists')
        index_list = [TEXT_SELECT]
        index_list.extend(lists[lists['Type'] == 'Index']['Name'].tolist())
        events_list = [TEXT_SELECT]
        events_list.extend(lists[lists['Type'] == 'Event']['Name'].tolist())

class ComparativeDataProcessor(DataProcessorBase):
    ds1_name = None
    ds2_name = None    

    def set_input_options(self, ds1, ds2):
        self.ds1_name = ds1
        self.ds2_name = ds2

    def get_data(self):
        merged_df = self.get_merged_data(self.ds1_name, self.ds2_name, {'Close_x': self.ds1_name, 'Close_y': self.ds2_name})
        
        filtered_df = merged_df
        filtered_df['Ratio'] = filtered_df[self.ds1_name] / filtered_df[self.ds2_name]

        self.add_growth_cols(filtered_df, open_price_col='Open_x', close_price_col=self.ds1_name, prefix=self.ds1_name)
        self.add_growth_cols(filtered_df, open_price_col='Open_y', close_price_col=self.ds2_name, prefix=self.ds2_name)
        
        return filtered_df

class VolatilityDataProcessor(DataProcessorBase):
    __ath_min_gap_days = None

    def set_input_options(self, ds_name, ath_min_gap_days, ath_max_gap_days):
        self.set_index_name(ds_name)
        self.__ath_min_gap_days = ath_min_gap_days
        self.__ath_max_gap_days = ath_max_gap_days

    def get_data(self):
        df = self.get_index_data()
        self.add_ath_cols(df, self.__ath_min_gap_days, self.__ath_max_gap_days)
        self.add_date_cat_cols(df)
        return df

class EventDataProcessor(DataProcessorBase):
    ds1_name = None
    event_name = None
    before_days = None
    after_days = None
    till_event_name = None
    chart_type = None

    def set_input_options(self, ds1, event_name, before_days, after_days, till_event_name, chart_type):
        self.ds1_name = ds1
        self.event_name = event_name
        self.before_days = before_days
        self.after_days = after_days
        self.till_event_name = till_event_name
        self.chart_type = chart_type

    def get_data(self):
        merged_df = self.get_merged_data(self.ds1_name, 'EventsInfo', how='left')
        merged_df['Events'] = merged_df['Events'].fillna('')
        
        df_list = []
        event_dates = []
        event_row_indexes = []
        
        if self.event_name == TEXT_SELECT:
            filtered_df = merged_df
        elif self.event_name == "All Time High":
            filtered_df = merged_df[merged_df['Is ATH']]
        else:
            filtered_df = merged_df[merged_df['Events'].str.contains(self.event_name)]
        
        if self.for_month is not None:
            filtered_df = filtered_df[filtered_df['Date'].dt.strftime("%B") == self.for_month]

        if self.for_day is not None:
            filtered_df = filtered_df[filtered_df['Date'].dt.strftime("%A") == self.for_day]
            
        event_row_indexes = filtered_df.index.tolist()

        for event_row_no in event_row_indexes:
            event_df = pd.DataFrame()
            
            for row_no in range(event_row_no - self.before_days, event_row_no + self.after_days + 1):
                if merged_df.index.contains(row_no):
                    event_df = event_df.append(merged_df.loc[row_no])
                    if self.till_event_name is not None and row_no >= event_row_no and self.till_event_name in merged_df.loc[row_no]['Events']:
                        break
            self.add_growth_cols(event_df, open_price_col='Open', close_price_col='Close')
            df_list.append(event_df)
            event_dates.append(merged_df.loc[event_row_no]['Date'])
        
        return df_list, event_dates

class DataVisualizerBase:
    input_controls = None
    __fig_counter = 0
    __figures = None
    data = None
    data_processor = None
    __layout = widgets.Layout(
            width='90%',
            grid_template_rows='auto auto auto',
            grid_template_columns='50% 50%'
        )
    __input_grid_cells = []

    def __init__(self):
        self.input_controls = {}

    def clear_input_filters(self):
        self.__input_grid_cells = []

    def show_for_month_filter_input(self):
        months_list = [TEXT_SELECT]
        months_list.extend(pd.date_range('2021-01-01','2021-12-31', 
              freq='MS').strftime("%B").tolist())
        input_control = widgets.Dropdown(
            options=months_list,
            description='For month',
            disabled=False
        )
        self.input_controls['for_month'] = input_control
        self.__input_grid_cells.append(input_control)
    
    def show_for_day_filter_input(self):
        days_list = [TEXT_SELECT]
        days_list.extend(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
        input_control = widgets.Dropdown(
            options=days_list,
            description='For day',
            #REVERT
            value='Friday',
            disabled=False
        )
        self.input_controls['for_day'] = input_control
        self.__input_grid_cells.append(input_control)

    def show_date_range_filter_input(self):
        input_control = widgets.DatePicker(
            description='From date',
            #REVERT
            #value = datetime.fromisoformat('2000-01-01'),
            value = datetime.fromisoformat('2021-10-19'),
            disabled=False
        )
        self.input_controls['from_date'] = input_control
        self.__input_grid_cells.append(input_control)

        input_control = widgets.DatePicker(
            description='To date',
            #REVERT
            #value = datetime.fromisoformat('2021-12-31'),
            value = datetime.fromisoformat('2022-06-16'),
            disabled=False
        )
        self.input_controls['to_date'] = input_control
        self.__input_grid_cells.append(input_control)

    def show_input_options(self, metadata):
        index_list_counter = 1

        if index_list is None:
            list_reader = ListReader()
            list_reader.read_lists()
        for meta in metadata:
            input_control = None

            if meta['type'] == 'index_list':
                input_control = widgets.Dropdown(
                    options=index_list,
                    value=index_list[index_list_counter],
                    description=meta['display_text'],
                    disabled=False,
                    layout = self.__layout
                )
                index_list_counter = index_list_counter + 1
            elif meta['type'] == 'events_list':
                selected_value = events_list[1]
                if 'default_value' in meta:
                    selected_value = meta['default_value']
                    
                input_control = widgets.Dropdown(
                    options=events_list,
                    value=selected_value,
                    description=meta['display_text'],
                    disabled=False
                )
            elif meta['type'] == 'chart_types':
                input_control = widgets.Dropdown(
                    options=[TEXT_CHART_TYPE_CANDLESTICK, TEXT_CHART_TYPE_GROWTH, TEXT_CHART_TYPE_GROWTH_PCT],
                    description=meta['display_text'],
                    disabled=False
                )
            elif meta['type'] == 'range':
                input_control = widgets.IntText(
                    min=meta['min'],
                    max=meta['max'],
                    description=meta['display_text'],
                    disabled=False
                )
                if meta['default_value'] is not None:
                    input_control.value = meta['default_value']
            elif meta['type'] == 'date':
                input_control = widgets.DatePicker(
                    description=meta['display_text'],
                    disabled=False
                )
        
            self.input_controls[meta['name']] = input_control
            self.__input_grid_cells.append(input_control)
        
        grid_box = widgets.GridBox(children=self.__input_grid_cells, layout=self.__layout)
    
        button_run = widgets.Button(
            description='Run',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Run'
        )

        button_run.on_click(self.__run)

        display(grid_box)
        display(button_run)
    
    def draw_scatterplot(self, index1, index2):
        self.__figures.append(px.scatter(x=self.data[index1], y=self.data[index2], title='Correlation between ' + index1 + ' & ' + index2))

    def draw_ratio_chart(self):
        self.__figures.append(px.line(self.data, x='Date', y='Ratio', title='Ratio'))

    def draw_line_chart(self, data, y, title=None):
        if title is None:
            title = y
        self.__figures.append(px.line(data, x='Date', y=y, title=title))

    def draw_growth_chart(self, index):
        self.__figures.append(px.line(self.data, x='Date', y=index + '_Growth', title=index + ' Growth'))

    def get_figures(self):
        return self.__figures

    def display_charts(self):
        plt.close()
        if self.__figures is not None:
            no_of_cols = 2
            fig = make_subplots(math.ceil(len(self.__figures) / no_of_cols), no_of_cols)
            for i in range(len(self.__figures)):
                self.__figures[i].show()

    def draw_candlestick_chart(self, df, marker_x_ref = None, annotation = None, marker_event_name = None):
        fig = go.Figure(data=[go.Candlestick(x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            xhoverformat='%a, %b %d, %Y',
        )])
        fig.update_layout(
            yaxis_tickformat = 'd',
            margin=dict(l=20, r=20, t=50, b=50),
            height=200,
            font_size=10
        )
        fig.update_xaxes(rangeslider_visible=False)
        shapes = []
        annotations = []

        if marker_event_name is not None:
            for marker_date in df[df['Events'].str.contains(marker_event_name)]['Date'].tolist():
                shapes.append(dict(
                        x0=marker_date, x1=marker_date, y0=0, y1=1, xref='x', yref='paper',
                        opacity=0.1,
                        line_width=10))
                annotations.append(dict(
                        x=marker_date, y=0.05, xref='x', yref='paper',
                        showarrow=False, xanchor='left', text=marker_event_name))

        if marker_x_ref is not None:
            shapes.append(dict(
                    x0=marker_x_ref, x1=marker_x_ref, y0=0, y1=1, xref='x', yref='paper',
                    opacity=0.1,
                    line_width=10))
            annotations.append(dict(
                    x=marker_x_ref, y=0.05, xref='x', yref='paper',
                    showarrow=False, xanchor='left', text=annotation))
            fig.update_layout(
                title = {
                'text': annotation + ' on ' + marker_x_ref.strftime("%a, %d %b, %Y"),
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=12)
                },
                shapes=shapes,
                annotations=annotations
            )
        self.__figures.append(fig)

    def reset_figure(self):
        self.__figures = []

    #REMOVE THIS
    def run(self):
        self.__run(None)

    def __run(self, btn):
        self.reset_figure()

        if self.input_controls['from_date'] is not None:
            self.data_processor.set_date_range_filter(self.input_controls['from_date'].value, self.input_controls['to_date'].value)
        
        if 'for_month' in self.input_controls:
            if self.input_controls['for_month'].value != TEXT_SELECT:
                self.data_processor.set_for_month_filter(self.input_controls['for_month'].value)
                
        if 'for_day' in self.input_controls:
            if self.input_controls['for_day'].value != TEXT_SELECT:
                self.data_processor.set_for_day_filter(self.input_controls['for_day'].value)

        print('Before start run_analysis')
        self.run_analysis()

        #display(Javascript('IPython.notebook.execute_cell_range(IPython.notebook.get_selected_index()+1, IPython.notebook.get_selected_index()+2)'))
        #plt.close()

class VolatilityDataVisualizer(DataVisualizerBase):
    __figure = None

    def show_input(self):
        self.clear_input_filters()
        #self.show_date_range_filter_input()
        self.show_input_options([{
            'type': 'index_list',
            'name': 'index1',
            'display_text': 'Index'
        }, {
            'type': 'range',
            'min': 30,
            'max': average_trading_days_in_year * 5,
            'default_value': average_trading_days_in_year,
            'name': 'athMinGapDays',
            'display_text': 'ATH Min Gap in Days'
        }, {
            'type': 'range',
            'min': average_trading_days_in_year,
            'max': average_trading_days_in_year * 10,
            'default_value': average_trading_days_in_year * 5,
            'name': 'athMaxGapDays',
            'display_text': 'ATH Max Gap in Days'
        }])
        self.data_processor = VolatilityDataProcessor()

    def run_analysis(self):
        print('run_analysis started')
        index_name = self.input_controls['index1'].value
        ath_min_gap_days = self.input_controls['athMinGapDays'].value
        ath_max_gap_days = self.input_controls['athMinGapDays'].value
        self.data_processor.set_input_options(index_name, ath_min_gap_days, ath_max_gap_days)
        
        df = self.data_processor.get_data()
        yearly_df = self.data_processor.get_yearly_data(df, ['green', 'indianred'])
        
        self.__figure = make_subplots(rows=3, cols=1)
        line_colors = ['indigo', 'brown', 'orange']
        for cycle in df['Cycle'].unique():
            self.__figure.append_trace(go.Scatter(
                    x=df[df.Cycle == cycle]['Date'],
                    y=df[df.Cycle == cycle]['Perc Returns Within Cycle'],
                    mode='lines', line_color=line_colors[cycle % 3]
                ), row=1, col=1)
            self.__figure.append_trace(go.Scatter(
                    x=df[df.Cycle == cycle]['Date'],
                    y=df[df.Cycle == cycle]['Close'],
                    mode='lines', line_color=line_colors[cycle % 3]
                ), row=2, col=1)

        self.__figure.append_trace(go.Bar(
                x=yearly_df['Year'], 
                y=yearly_df['Growth_Perc']
            ), 
            row=3, col=1)

    def display_charts(self):
        self.__figure.show()

class ComparativeDataVisualizer(DataVisualizerBase):
    def show_input(self):
        self.clear_input_filters()
        self.show_date_range_filter_input()
        self.show_input_options([{
            'type': 'index_list',
            'name': 'index1',
            'display_text': 'Index 1'
        }, {
            'type': 'index_list',
            'name': 'index2',
            'display_text': 'Index 2'
        }])
        self.data_processor = ComparativeDataProcessor()
        
    def run_analysis(self):
        ds1_name = self.input_controls['index1'].value
        ds2_name = self.input_controls['index2'].value

        self.data_processor.set_input_options(ds1_name, ds2_name)
        self.data = self.data_processor.get_data()
        
        self.draw_scatterplot(ds1_name, ds2_name)
        self.draw_ratio_chart()
        self.draw_growth_chart(ds1_name)
        self.draw_growth_chart(ds2_name)
        
class EventDataVisualizer(DataVisualizerBase):
    def show_input(self):
        self.clear_input_filters()
        self.show_date_range_filter_input()
        self.show_for_month_filter_input()
        self.show_for_day_filter_input()
        self.show_input_options([{
            'type': 'index_list',
            'name': 'index',
            'display_text': 'Index'
        }, {
            'type': 'events_list',
            'name': 'event',
            'default_value': TEXT_SELECT,
            'display_text': 'Event'
        }, {
            'type': 'events_list',
            'name': 'marker_event_name',
            'default_value': TEXT_SELECT,
            'display_text': 'Mark event'
        }, {
            'type': 'range',
            'min': 0,
            'max': 50,
            #REVERT
            #'default_value': 10,
            'default_value': 2,
            'name': 'before_days',
            'display_text': 'Before days'
        }, {
            'type': 'range',
            'min': 0,
            'max': 50,
            #REVERT
            #'default_value': 10,
            'default_value': 5,
            'name': 'after_days',
            'display_text': 'After days'
        }, {
            'type': 'events_list',
            'name': 'till_event_name',
            'default_value': TEXT_SELECT,
            'display_text': 'Till event'
        }, {
            'type': 'chart_types',
            'name': 'chart_type',
            'display_text': 'Chart type'
        }])
        self.data_processor = EventDataProcessor()

    def run_analysis(self):
        index_name = self.input_controls['index'].value
        event_name = self.input_controls['event'].value
        before_days = self.input_controls['before_days'].value
        after_days = self.input_controls['after_days'].value
        marker_event_name = self.input_controls['marker_event_name'].value
        till_event_name = self.input_controls['till_event_name'].value
        chart_type = self.input_controls['chart_type'].value

        if marker_event_name == TEXT_SELECT:
            marker_event_name = None

        self.data_processor.set_input_options(index_name, event_name, before_days, after_days, till_event_name, None)
        
        self.data, event_dates = self.data_processor.get_data()
        if chart_type == TEXT_CHART_TYPE_GROWTH:
            for i in range(len(self.data)):
                self.draw_line_chart(self.data[i], y='Growth', title=event_name + ' on ' + event_dates[i].strftime("%a, %d %b, %Y"))
        elif chart_type == TEXT_CHART_TYPE_GROWTH_PCT:
            for i in range(len(self.data)):
                self.draw_line_chart(self.data[i], y='Growth_Pct', title=event_name + ' on ' + event_dates[i].strftime("%a, %d %b, %Y"))
        else:
            for i in range(len(self.data)):
                self.draw_candlestick_chart(self.data[i], event_dates[i], event_name, marker_event_name)
            
