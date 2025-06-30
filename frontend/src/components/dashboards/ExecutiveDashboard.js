import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
         BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { MapContainer, TileLayer, Marker, Popup, HeatmapLayer } from 'react-leaflet';
import { useApi } from '../../hooks/useApi';
import { formatCurrency, formatNumber } from '../../utils/formatters';

const ExecutiveDashboard = () => {
  const [timeRange, setTimeRange] = useState('30d');
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const { apiCall } = useApi();

  useEffect(() => {
    loadDashboardData();
  }, [timeRange]);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const [salesData, geoData, performanceData] = await Promise.all([
        apiCall('/analytics/sales-performance', { 
          params: { time_range: timeRange } 
        }),
        apiCall('/analytics/geographic-heatmap', { 
          params: { time_range: timeRange } 
        }),
        apiCall('/analytics/dealer-performance', { 
          params: { time_range: timeRange } 
        })
      ]);

      setDashboardData({
        sales: salesData,
        geographic: geoData,
        performance: performanceData
      });
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const KPICard = ({ title, value, change, icon, color = "blue" }) => (
    <Card className={`border-l-4 border-l-${color}-500`}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-gray-600">{title}</p>
            <p className="text-2xl font-bold text-gray-900">{value}</p>
            <p className={`text-sm ${change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {change >= 0 ? '↗' : '↘'} {Math.abs(change)}% from last period
            </p>
          </div>
          <div className={`text-${color}-500 text-3xl`}>
            {icon}
          </div>
        </div>
      </CardContent>
    </Card>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">Executive Dashboard</h1>
        <select 
          value={timeRange} 
          onChange={(e) => setTimeRange(e.target.value)}
          className="px-4 py-2 border rounded-lg"
        >
          <option value="7d">Last 7 Days</option>
          <option value="30d">Last 30 Days</option>
          <option value="90d">Last 90 Days</option>
          <option value="1y">Last Year</option>
        </select>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard
          title="Total Sales"
          value={formatCurrency(dashboardData?.sales?.total_sales || 0)}
          change={dashboardData?.sales?.sales_growth || 0}
          icon="💰"
          color="green"
        />
        <KPICard
          title="Total Orders"
          value={formatNumber(dashboardData?.sales?.total_orders || 0)}
          change={dashboardData?.sales?.order_growth || 0}
          icon="📦"
          color="blue"
        />
        <KPICard
          title="Active Dealers"
          value={formatNumber(dashboardData?.performance?.active_dealers || 0)}
          change={dashboardData?.performance?.dealer_growth || 0}
          icon="👥"
          color="purple"
        />
        <KPICard
          title="Avg Order Value"
          value={formatCurrency(dashboardData?.sales?.avg_order_value || 0)}
          change={dashboardData?.sales?.aov_growth || 0}
          icon="📈"
          color="orange"
        />
      </div>

      
      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sales Trend */}
        <Card>
          <CardHeader>
            <CardTitle>Sales Trend</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={dashboardData?.sales?.daily_sales || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip formatter={(value) => [formatCurrency(value), 'Sales']} />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="sales" 
                  stroke="#2563eb" 
                  strokeWidth={2}
                  dot={{ fill: '#2563eb' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Territory Performance */}
        <Card>
          <CardHeader>
            <CardTitle>Territory Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={dashboardData?.sales?.territory_performance || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="territory" />
                <YAxis />
                <Tooltip formatter={(value) => [formatCurrency(value), 'Sales']} />
                <Bar dataKey="sales" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Second Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Product Categories */}
        <Card>
          <CardHeader>
            <CardTitle>Sales by Product Category</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={dashboardData?.product_categories || []}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {dashboardData?.product_categories?.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [`${value}%`, 'Share']} />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Top Dealers */}
        <Card>
          <CardHeader>
            <CardTitle>Top Performing Dealers</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {dashboardData?.performance?.top_dealers?.map((dealer, index) => (
                <div key={dealer.name} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">
                      {index + 1}
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">{dealer.name}</p>
                      <p className="text-sm text-gray-600">{dealer.orders} orders</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-bold text-green-600">{formatCurrency(dealer.sales)}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Geographic Heatmap */}
      <Card>
        <CardHeader>
          <CardTitle>Geographic Sales Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96 bg-gray-200 rounded-lg flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl mb-2">🗺️</div>
              <p className="text-gray-600">Interactive map would be displayed here</p>
              <p className="text-sm text-gray-500 mt-2">
                Showing sales data across {dashboardData?.geographic?.heatmap_data?.length || 0} locations
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardContent className="p-6 text-center">
            <div className="text-3xl mb-2">📊</div>
            <p className="text-2xl font-bold text-gray-900">
              {formatNumber(dashboardData?.sales?.total_orders || 0)}
            </p>
            <p className="text-sm text-gray-600">Total Orders This Period</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6 text-center">
            <div className="text-3xl mb-2">🎯</div>
            <p className="text-2xl font-bold text-gray-900">94.2%</p>
            <p className="text-sm text-gray-600">Customer Satisfaction</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6 text-center">
            <div className="text-3xl mb-2">⚡</div>
            <p className="text-2xl font-bold text-gray-900">2.3 days</p>
            <p className="text-sm text-gray-600">Avg Processing Time</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ExecutiveDashboard;