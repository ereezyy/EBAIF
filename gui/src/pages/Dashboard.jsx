import React from 'react'
import { 
  Grid, 
  Card, 
  CardContent, 
  Typography, 
  Box,
  LinearProgress,
  Chip
} from '@mui/material'
import { motion } from 'framer-motion'
import {
  SmartToy,
  TrendingUp,
  Speed,
  Psychology,
  Computer,
  NetworkCheck
} from '@mui/icons-material'
import AgentOverview from '../components/AgentOverview'
import PerformanceChart from '../components/PerformanceChart'
import RealtimeStats from '../components/RealtimeStats'
import TaskQueue from '../components/TaskQueue'
import NetworkVisualization from '../components/NetworkVisualization'

const Dashboard = () => {
  const stats = [
    {
      title: 'Active Agents',
      value: '5',
      change: '+2',
      icon: SmartToy,
      color: '#667eea'
    },
    {
      title: 'Tasks Completed',
      value: '1,247',
      change: '+89',
      icon: TrendingUp,
      color: '#764ba2'
    },
    {
      title: 'System Efficiency',
      value: '94.2%',
      change: '+5.3%',
      icon: Speed,
      color: '#4ecdc4'
    },
    {
      title: 'AI Processing',
      value: '847 req/min',
      change: '+12%',
      icon: Psychology,
      color: '#ff6b6b'
    }
  ]

  return (
    <Box sx={{ p: 3 }}>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Typography variant="h3" sx={{ color: 'white', mb: 3, fontWeight: 700 }}>
          🤖 Agentic AI Dashboard
        </Typography>
      </motion.div>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {stats.map((stat, index) => (
          <Grid item xs={12} sm={6} md={3} key={stat.title}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card sx={{ 
                height: '100%',
                background: 'rgba(26, 31, 46, 0.8)',
                backdropFilter: 'blur(10px)',
                border: `1px solid ${stat.color}40`,
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  border: `1px solid ${stat.color}80`,
                  boxShadow: `0 8px 32px ${stat.color}20`
                }
              }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Box sx={{ 
                      p: 1, 
                      borderRadius: '12px', 
                      background: `${stat.color}20`,
                      mr: 2
                    }}>
                      <stat.icon sx={{ color: stat.color }} />
                    </Box>
                    <Box>
                      <Typography variant="h4" sx={{ color: 'white', fontWeight: 700 }}>
                        {stat.value}
                      </Typography>
                      <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                        {stat.title}
                      </Typography>
                    </Box>
                  </Box>
                  <Chip 
                    label={stat.change}
                    size="small"
                    sx={{ 
                      background: `${stat.color}20`,
                      color: stat.color,
                      fontWeight: 600
                    }}
                  />
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        ))}
      </Grid>

      {/* Main Content Grid */}
      <Grid container spacing={3}>
        {/* Agent Overview */}
        <Grid item xs={12} lg={8}>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <AgentOverview />
          </motion.div>
        </Grid>

        {/* Real-time Stats */}
        <Grid item xs={12} lg={4}>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <RealtimeStats />
          </motion.div>
        </Grid>

        {/* Performance Chart */}
        <Grid item xs={12} lg={8}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <PerformanceChart />
          </motion.div>
        </Grid>

        {/* Task Queue */}
        <Grid item xs={12} lg={4}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
          >
            <TaskQueue />
          </motion.div>
        </Grid>

        {/* Network Visualization */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            <NetworkVisualization />
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  )
}

export default Dashboard