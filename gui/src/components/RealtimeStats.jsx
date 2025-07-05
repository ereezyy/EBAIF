import React, { useState, useEffect } from 'react'
import { 
  Card, 
  CardContent, 
  Typography, 
  Box,
  LinearProgress,
  Chip
} from '@mui/material'
import { 
  Memory, 
  Storage, 
  NetworkCheck, 
  Speed,
  TrendingUp
} from '@mui/icons-material'
import { motion } from 'framer-motion'

const RealtimeStats = () => {
  const [stats, setStats] = useState({
    cpuUsage: 23.4,
    memoryUsage: 67.8,
    networkLatency: 12.3,
    apiCalls: 847,
    responseTime: 156
  })

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setStats(prev => ({
        cpuUsage: Math.max(10, Math.min(95, prev.cpuUsage + (Math.random() - 0.5) * 10)),
        memoryUsage: Math.max(20, Math.min(90, prev.memoryUsage + (Math.random() - 0.5) * 5)),
        networkLatency: Math.max(5, Math.min(100, prev.networkLatency + (Math.random() - 0.5) * 20)),
        apiCalls: prev.apiCalls + Math.floor(Math.random() * 10),
        responseTime: Math.max(50, Math.min(500, prev.responseTime + (Math.random() - 0.5) * 50))
      }))
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const getProgressColor = (value, thresholds = [50, 80]) => {
    if (value < thresholds[0]) return '#4ecdc4'
    if (value < thresholds[1]) return '#ffa726'
    return '#ff6b6b'
  }

  const metrics = [
    {
      label: 'CPU Usage',
      value: stats.cpuUsage,
      suffix: '%',
      icon: Speed,
      color: getProgressColor(stats.cpuUsage)
    },
    {
      label: 'Memory Usage',
      value: stats.memoryUsage,
      suffix: '%',
      icon: Memory,
      color: getProgressColor(stats.memoryUsage)
    },
    {
      label: 'Network Latency',
      value: stats.networkLatency,
      suffix: 'ms',
      icon: NetworkCheck,
      color: getProgressColor(stats.networkLatency, [20, 50])
    },
    {
      label: 'Storage Usage',
      value: 34.2,
      suffix: '%',
      icon: Storage,
      color: '#4ecdc4'
    }
  ]

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h5" sx={{ color: 'white', mb: 3, fontWeight: 600 }}>
          ⚡ Real-time Stats
        </Typography>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {metrics.map((metric, index) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <Box>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <metric.icon sx={{ color: metric.color, mr: 1, fontSize: 20 }} />
                  <Typography variant="body2" sx={{ color: '#b8c5d6', flexGrow: 1 }}>
                    {metric.label}
                  </Typography>
                  <Typography variant="body2" sx={{ color: 'white', fontWeight: 600 }}>
                    {metric.value.toFixed(1)}{metric.suffix}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={metric.value}
                  sx={{
                    height: 6,
                    borderRadius: 3,
                    background: 'rgba(255, 255, 255, 0.1)',
                    '& .MuiLinearProgress-bar': {
                      background: `linear-gradient(90deg, ${metric.color}80, ${metric.color})`
                    }
                  }}
                />
              </Box>
            </motion.div>
          ))}

          <Box sx={{ mt: 2, p: 2, borderRadius: '12px', background: 'rgba(76, 175, 80, 0.1)' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <TrendingUp sx={{ color: '#4caf50', mr: 1 }} />
              <Typography variant="body2" sx={{ color: 'white', fontWeight: 600 }}>
                API Performance
              </Typography>
            </Box>
            <Typography variant="h6" sx={{ color: '#4caf50', fontWeight: 700 }}>
              {stats.apiCalls.toLocaleString()} req/min
            </Typography>
            <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
              Avg Response: {stats.responseTime.toFixed(0)}ms
            </Typography>
          </Box>

          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Chip
              label="All Systems Operational"
              size="small"
              sx={{
                background: 'rgba(76, 175, 80, 0.2)',
                color: '#4caf50',
                fontWeight: 600
              }}
            />
            <Chip
              label="Auto-scaling Active"
              size="small"
              sx={{
                background: 'rgba(102, 126, 234, 0.2)',
                color: '#667eea',
                fontWeight: 600
              }}
            />
          </Box>
        </Box>
      </CardContent>
    </Card>
  )
}

export default RealtimeStats