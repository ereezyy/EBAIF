import React from 'react'
import { Card, CardContent, Typography, Box } from '@mui/material'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'

const PerformanceChart = () => {
  const data = [
    { time: '00:00', efficiency: 85, tasks: 12, agents: 4 },
    { time: '04:00', efficiency: 89, tasks: 18, agents: 4 },
    { time: '08:00', efficiency: 94, tasks: 25, agents: 5 },
    { time: '12:00', efficiency: 96, tasks: 32, agents: 5 },
    { time: '16:00', efficiency: 92, tasks: 28, agents: 5 },
    { time: '20:00', efficiency: 88, tasks: 22, agents: 4 },
    { time: '24:00', efficiency: 91, tasks: 19, agents: 4 }
  ]

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <Box sx={{ 
          background: 'rgba(26, 31, 46, 0.95)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: '8px',
          p: 2
        }}>
          <Typography variant="body2" sx={{ color: 'white', mb: 1 }}>
            Time: {label}
          </Typography>
          {payload.map((entry, index) => (
            <Typography key={index} variant="body2" sx={{ color: entry.color }}>
              {entry.dataKey}: {entry.value}
              {entry.dataKey === 'efficiency' ? '%' : ''}
            </Typography>
          ))}
        </Box>
      )
    }
    return null
  }

  return (
    <Card sx={{ height: '400px' }}>
      <CardContent>
        <Typography variant="h5" sx={{ color: 'white', mb: 3, fontWeight: 600 }}>
          📊 System Performance
        </Typography>
        
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="efficiencyGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#667eea" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#667eea" stopOpacity={0.1}/>
              </linearGradient>
              <linearGradient id="tasksGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#4ecdc4" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#4ecdc4" stopOpacity={0.1}/>
              </linearGradient>
            </defs>
            
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
            <XAxis 
              dataKey="time" 
              stroke="#b8c5d6"
              fontSize={12}
            />
            <YAxis 
              stroke="#b8c5d6"
              fontSize={12}
            />
            <Tooltip content={<CustomTooltip />} />
            
            <Area
              type="monotone"
              dataKey="efficiency"
              stroke="#667eea"
              strokeWidth={2}
              fill="url(#efficiencyGradient)"
              name="Efficiency (%)"
            />
            
            <Line
              type="monotone"
              dataKey="tasks"
              stroke="#4ecdc4"
              strokeWidth={2}
              dot={{ fill: '#4ecdc4', strokeWidth: 2, r: 4 }}
              name="Tasks Completed"
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

export default PerformanceChart