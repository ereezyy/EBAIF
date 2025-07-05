import React from 'react'
import { 
  Grid, 
  Card, 
  CardContent, 
  Typography, 
  Box,
  LinearProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  ButtonGroup
} from '@mui/material'
import { motion } from 'framer-motion'
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell
} from 'recharts'

const Analytics = () => {
  const [timeRange, setTimeRange] = React.useState('7days')

  // Sample data for charts
  const taskPerformanceData = [
    { day: 'Mon', research: 45, code: 32, communication: 28, data: 15 },
    { day: 'Tue', research: 50, code: 38, communication: 32, data: 20 },
    { day: 'Wed', research: 42, code: 40, communication: 35, data: 22 },
    { day: 'Thu', research: 48, code: 45, communication: 30, data: 25 },
    { day: 'Fri', research: 55, code: 48, communication: 40, data: 30 },
    { day: 'Sat', research: 60, code: 50, communication: 45, data: 28 },
    { day: 'Sun', research: 58, code: 52, communication: 50, data: 35 },
  ]

  const systemPerformanceData = [
    { time: '00:00', cpu: 23, memory: 45, network: 32 },
    { time: '04:00', cpu: 32, memory: 48, network: 28 },
    { time: '08:00', cpu: 45, memory: 56, network: 35 },
    { time: '12:00', cpu: 65, memory: 72, network: 45 },
    { time: '16:00', cpu: 52, memory: 68, network: 42 },
    { time: '20:00', cpu: 38, memory: 52, network: 30 },
    { time: '24:00', cpu: 25, memory: 48, network: 25 }
  ]

  const resourceUsageData = [
    { name: 'Research', value: 35 },
    { name: 'Code Editing', value: 25 },
    { name: 'Communication', value: 20 },
    { name: 'Data Analysis', value: 15 },
    { name: 'System Operations', value: 5 }
  ]

  const COLORS = ['#667eea', '#4ecdc4', '#ffa726', '#ff6b6b', '#95a5a6']

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <Box sx={{ 
          background: 'rgba(26, 31, 46, 0.95)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: '8px',
          p: 2,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.2)'
        }}>
          <Typography variant="body2" sx={{ color: 'white', mb: 1 }}>
            {label}
          </Typography>
          {payload.map((entry, index) => (
            <Typography key={`item-${index}`} variant="body2" sx={{ color: entry.color }}>
              {`${entry.name}: ${entry.value}`}
              {entry.name === 'cpu' || entry.name === 'memory' || entry.name === 'network' ? '%' : ''}
            </Typography>
          ))}
        </Box>
      )
    }
    return null
  }

  return (
    <Box sx={{ p: 3 }}>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
          <Typography variant="h3" sx={{ color: 'white', fontWeight: 700 }}>
            📊 System Analytics
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <ButtonGroup 
              variant="outlined" 
              sx={{ 
                mr: 2,
                '& .MuiButton-root': {
                  color: '#b8c5d6',
                  borderColor: 'rgba(255, 255, 255, 0.1)'
                },
                '& .MuiButton-root.Mui-selected': {
                  color: '#667eea',
                  borderColor: '#667eea',
                  background: 'rgba(102, 126, 234, 0.1)'
                }
              }}
            >
              <Button 
                variant={timeRange === '1day' ? 'contained' : 'outlined'} 
                onClick={() => setTimeRange('1day')}
                sx={{ 
                  background: timeRange === '1day' ? 'rgba(102, 126, 234, 0.2)' : 'transparent',
                  color: timeRange === '1day' ? '#667eea' : '#b8c5d6'
                }}
              >
                24h
              </Button>
              <Button 
                variant={timeRange === '7days' ? 'contained' : 'outlined'} 
                onClick={() => setTimeRange('7days')}
                sx={{ 
                  background: timeRange === '7days' ? 'rgba(102, 126, 234, 0.2)' : 'transparent',
                  color: timeRange === '7days' ? '#667eea' : '#b8c5d6'
                }}
              >
                7d
              </Button>
              <Button 
                variant={timeRange === '30days' ? 'contained' : 'outlined'} 
                onClick={() => setTimeRange('30days')}
                sx={{ 
                  background: timeRange === '30days' ? 'rgba(102, 126, 234, 0.2)' : 'transparent',
                  color: timeRange === '30days' ? '#667eea' : '#b8c5d6'
                }}
              >
                30d
              </Button>
            </ButtonGroup>
            <Button
              variant="contained"
              sx={{
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                borderRadius: '12px',
                px: 3
              }}
            >
              Download Report
            </Button>
          </Box>
        </Box>
      </motion.div>

      <Grid container spacing={3}>
        {/* Agent Task Performance */}
        <Grid item xs={12} lg={8}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Typography variant="h5" sx={{ color: 'white', fontWeight: 600 }}>
                    Agent Task Performance
                  </Typography>
                  <FormControl 
                    size="small"
                    sx={{ 
                      minWidth: 150,
                      '& .MuiOutlinedInput-root': {
                        borderRadius: '8px',
                        backgroundColor: 'rgba(255, 255, 255, 0.05)'
                      },
                      '& .MuiSelect-select': {
                        color: 'white'
                      },
                      '& .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'rgba(255, 255, 255, 0.1)'
                      }
                    }}
                  >
                    <Select
                      value="tasks"
                      displayEmpty
                    >
                      <MenuItem value="tasks">Tasks Completed</MenuItem>
                      <MenuItem value="time">Time Spent</MenuItem>
                      <MenuItem value="efficiency">Efficiency</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={taskPerformanceData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                    <XAxis dataKey="day" stroke="#b8c5d6" />
                    <YAxis stroke="#b8c5d6" />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="research" name="Research" stackId="a" fill="#667eea" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="code" name="Code" stackId="a" fill="#4ecdc4" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="communication" name="Communication" stackId="a" fill="#ffa726" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="data" name="Data Analysis" stackId="a" fill="#ff6b6b" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2, gap: 4 }}>
                  {[
                    { name: 'Research', color: '#667eea' },
                    { name: 'Code', color: '#4ecdc4' },
                    { name: 'Communication', color: '#ffa726' },
                    { name: 'Data Analysis', color: '#ff6b6b' },
                  ].map((item) => (
                    <Box key={item.name} sx={{ display: 'flex', alignItems: 'center' }}>
                      <Box sx={{ 
                        width: 12, 
                        height: 12, 
                        borderRadius: '4px', 
                        background: item.color,
                        mr: 1 
                      }} />
                      <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                        {item.name}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        {/* Resource Usage */}
        <Grid item xs={12} lg={4}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h5" sx={{ color: 'white', mb: 3, fontWeight: 600 }}>
                  Resource Allocation
                </Typography>
                <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={resourceUsageData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={100}
                        innerRadius={60}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {resourceUsageData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip content={<CustomTooltip />} />
                    </PieChart>
                  </ResponsiveContainer>
                </Box>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 2, mt: 2 }}>
                  {resourceUsageData.map((entry, index) => (
                    <Box key={entry.name} sx={{ display: 'flex', alignItems: 'center' }}>
                      <Box sx={{ 
                        width: 12, 
                        height: 12, 
                        borderRadius: '4px', 
                        background: COLORS[index % COLORS.length],
                        mr: 1 
                      }} />
                      <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                        {entry.name}: {entry.value}%
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* System Performance */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ color: 'white', mb: 3, fontWeight: 600 }}>
                  System Performance Metrics
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={systemPerformanceData}>
                    <defs>
                      <linearGradient id="cpuColor" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#667eea" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#667eea" stopOpacity={0}/>
                      </linearGradient>
                      <linearGradient id="memoryColor" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#4ecdc4" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#4ecdc4" stopOpacity={0}/>
                      </linearGradient>
                      <linearGradient id="networkColor" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#ffa726" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#ffa726" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                    <XAxis dataKey="time" stroke="#b8c5d6" />
                    <YAxis stroke="#b8c5d6" />
                    <Tooltip content={<CustomTooltip />} />
                    <Area 
                      type="monotone" 
                      dataKey="cpu" 
                      name="CPU Usage" 
                      stroke="#667eea" 
                      fillOpacity={1} 
                      fill="url(#cpuColor)" 
                    />
                    <Area 
                      type="monotone" 
                      dataKey="memory" 
                      name="Memory Usage" 
                      stroke="#4ecdc4" 
                      fillOpacity={1} 
                      fill="url(#memoryColor)" 
                    />
                    <Area 
                      type="monotone" 
                      dataKey="network" 
                      name="Network" 
                      stroke="#ffa726" 
                      fillOpacity={1} 
                      fill="url(#networkColor)" 
                    />
                  </AreaChart>
                </ResponsiveContainer>
                <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2, gap: 4 }}>
                  {[
                    { name: 'CPU Usage', color: '#667eea' },
                    { name: 'Memory Usage', color: '#4ecdc4' },
                    { name: 'Network Activity', color: '#ffa726' },
                  ].map((item) => (
                    <Box key={item.name} sx={{ display: 'flex', alignItems: 'center' }}>
                      <Box sx={{ 
                        width: 12, 
                        height: 12, 
                        borderRadius: '4px', 
                        background: item.color,
                        mr: 1 
                      }} />
                      <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                        {item.name}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Key Performance Metrics */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ color: 'white', mb: 4, fontWeight: 600 }}>
                  Key Performance Metrics
                </Typography>
                
                <Grid container spacing={4}>
                  {[
                    { 
                      name: 'Agent Efficiency',
                      current: 92.7,
                      target: 95.0,
                      change: '+1.5%',
                      color: '#667eea'
                    },
                    { 
                      name: 'Task Completion Rate',
                      current: 97.3,
                      target: 99.0,
                      change: '+0.8%',
                      color: '#4ecdc4'
                    },
                    { 
                      name: 'Response Time (ms)',
                      current: 125,
                      target: 100,
                      change: '-15ms',
                      color: '#ffa726'
                    },
                    { 
                      name: 'Resource Utilization',
                      current: 67.4,
                      target: 75.0,
                      change: '+2.3%',
                      color: '#ff6b6b'
                    },
                  ].map((metric) => (
                    <Grid item xs={12} sm={6} md={3} key={metric.name}>
                      <Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                          <Typography variant="body1" sx={{ color: 'white', fontWeight: 600 }}>
                            {metric.name}
                          </Typography>
                          <Typography 
                            variant="body2" 
                            sx={{ 
                              color: metric.change.startsWith('+') ? '#4caf50' : '#ff6b6b',
                              fontWeight: 600
                            }}
                          >
                            {metric.change}
                          </Typography>
                        </Box>
                        
                        <Box sx={{ mb: 1 }}>
                          <Typography variant="h4" sx={{ color: 'white', fontWeight: 700 }}>
                            {metric.current}{metric.name.includes('ms') ? '' : '%'}
                          </Typography>
                          <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                            Target: {metric.target}{metric.name.includes('ms') ? '' : '%'}
                          </Typography>
                        </Box>
                        
                        <LinearProgress
                          variant="determinate"
                          value={(metric.current / metric.target) * 100}
                          sx={{
                            height: 8,
                            borderRadius: 4,
                            background: 'rgba(255, 255, 255, 0.1)',
                            '& .MuiLinearProgress-bar': {
                              background: `linear-gradient(90deg, ${metric.color}80, ${metric.color})`
                            }
                          }}
                        />
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  )
}

export default Analytics