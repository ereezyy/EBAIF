import React from 'react'
import { 
  Grid, 
  Card, 
  CardContent, 
  Typography, 
  Box,
  Button,
  IconButton,
  Chip,
  Avatar
} from '@mui/material'
import { 
  Add, 
  PlayArrow, 
  Pause, 
  Stop, 
  Settings,
  SmartToy,
  TrendingUp,
  Memory,
  Speed
} from '@mui/icons-material'
import { motion } from 'framer-motion'

const Agents = () => {
  const agents = [
    {
      id: 'agent_1',
      name: 'Research Assistant',
      status: 'active',
      uptime: '2h 45m',
      tasksCompleted: 47,
      efficiency: 94.2,
      memory: 67.8,
      cpu: 23.4,
      capabilities: ['Web Browsing', 'AI Analysis', 'Research', 'Data Mining'],
      lastActivity: 'Analyzing market trends for Q4 report'
    },
    {
      id: 'agent_2',
      name: 'Communication Hub',
      status: 'active',
      uptime: '5h 12m',
      tasksCompleted: 89,
      efficiency: 98.7,
      memory: 45.2,
      cpu: 12.1,
      capabilities: ['Email Management', 'SMS', 'Social Media', 'Voice Calls'],
      lastActivity: 'Processing 12 new emails and 3 SMS messages'
    },
    {
      id: 'agent_3',
      name: 'Code Editor',
      status: 'busy',
      uptime: '1h 33m',
      tasksCompleted: 23,
      efficiency: 87.3,
      memory: 78.9,
      cpu: 65.7,
      capabilities: ['Code Generation', 'Debugging', 'Testing', 'Refactoring'],
      lastActivity: 'Refactoring authentication module'
    },
    {
      id: 'agent_4',
      name: 'Data Analyst',
      status: 'idle',
      uptime: '3h 18m',
      tasksCompleted: 56,
      efficiency: 91.5,
      memory: 34.1,
      cpu: 8.2,
      capabilities: ['Data Processing', 'Analytics', 'Reporting', 'Visualization'],
      lastActivity: 'Generated quarterly performance report'
    },
    {
      id: 'agent_5',
      name: 'Web Browser',
      status: 'active',
      uptime: '4h 07m',
      tasksCompleted: 134,
      efficiency: 96.1,
      memory: 52.3,
      cpu: 19.6,
      capabilities: ['Web Scraping', 'Content Analysis', 'Search', 'Monitoring'],
      lastActivity: 'Monitoring competitor pricing updates'
    },
    {
      id: 'agent_6',
      name: 'Multi-Modal AI',
      status: 'paused',
      uptime: '0h 00m',
      tasksCompleted: 0,
      efficiency: 0,
      memory: 0,
      cpu: 0,
      capabilities: ['Text Analysis', 'Image Processing', 'Voice Recognition', 'Translation'],
      lastActivity: 'Agent paused for maintenance'
    }
  ]

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return '#4ecdc4'
      case 'busy': return '#ffa726'
      case 'idle': return '#95a5a6'
      case 'paused': return '#ff6b6b'
      default: return '#667eea'
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'active': 
      case 'busy': return <PlayArrow />
      case 'idle': return <Pause />
      case 'paused': return <Stop />
      default: return <SmartToy />
    }
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
            🤖 Agent Management
          </Typography>
          <Button
            variant="contained"
            startIcon={<Add />}
            sx={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              borderRadius: '12px',
              px: 3
            }}
          >
            Create New Agent
          </Button>
        </Box>
      </motion.div>

      <Grid container spacing={3}>
        {agents.map((agent, index) => (
          <Grid item xs={12} md={6} key={agent.id}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card sx={{ 
                height: '100%',
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: `0 8px 32px ${getStatusColor(agent.status)}20`
                }
              }}>
                <CardContent sx={{ p: 3 }}>
                  {/* Header */}
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                    <Avatar sx={{ 
                      background: `${getStatusColor(agent.status)}20`,
                      color: getStatusColor(agent.status),
                      mr: 2,
                      width: 56,
                      height: 56
                    }}>
                      <SmartToy sx={{ fontSize: 28 }} />
                    </Avatar>
                    
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="h5" sx={{ color: 'white', fontWeight: 600, mb: 1 }}>
                        {agent.name}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Chip
                          icon={getStatusIcon(agent.status)}
                          label={agent.status.toUpperCase()}
                          size="small"
                          sx={{
                            background: `${getStatusColor(agent.status)}20`,
                            color: getStatusColor(agent.status),
                            fontWeight: 600
                          }}
                        />
                        <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                          Uptime: {agent.uptime}
                        </Typography>
                      </Box>
                    </Box>

                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <IconButton size="small" sx={{ color: '#4ecdc4' }}>
                        <PlayArrow />
                      </IconButton>
                      <IconButton size="small" sx={{ color: '#ffa726' }}>
                        <Pause />
                      </IconButton>
                      <IconButton size="small" sx={{ color: '#b8c5d6' }}>
                        <Settings />
                      </IconButton>
                    </Box>
                  </Box>

                  {/* Stats */}
                  <Grid container spacing={2} sx={{ mb: 3 }}>
                    <Grid item xs={6}>
                      <Box sx={{ textAlign: 'center', p: 2, borderRadius: '8px', background: 'rgba(255, 255, 255, 0.05)' }}>
                        <TrendingUp sx={{ color: '#4ecdc4', mb: 1 }} />
                        <Typography variant="h6" sx={{ color: 'white', fontWeight: 600 }}>
                          {agent.tasksCompleted}
                        </Typography>
                        <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                          Tasks Completed
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6}>
                      <Box sx={{ textAlign: 'center', p: 2, borderRadius: '8px', background: 'rgba(255, 255, 255, 0.05)' }}>
                        <Speed sx={{ color: '#667eea', mb: 1 }} />
                        <Typography variant="h6" sx={{ color: 'white', fontWeight: 600 }}>
                          {agent.efficiency}%
                        </Typography>
                        <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                          Efficiency
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>

                  {/* Resource Usage */}
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" sx={{ color: 'white', mb: 2, fontWeight: 600 }}>
                      Resource Usage
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 2 }}>
                      <Box sx={{ flexGrow: 1 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                          <Typography variant="body2" sx={{ color: '#b8c5d6', fontSize: '0.8rem' }}>
                            Memory
                          </Typography>
                          <Typography variant="body2" sx={{ color: 'white', fontSize: '0.8rem' }}>
                            {agent.memory}%
                          </Typography>
                        </Box>
                        <Box sx={{ 
                          height: 4, 
                          background: 'rgba(255, 255, 255, 0.1)', 
                          borderRadius: 2,
                          overflow: 'hidden'
                        }}>
                          <Box sx={{ 
                            width: `${agent.memory}%`, 
                            height: '100%', 
                            background: agent.memory > 80 ? '#ff6b6b' : agent.memory > 60 ? '#ffa726' : '#4ecdc4',
                            transition: 'width 0.3s ease'
                          }} />
                        </Box>
                      </Box>
                      <Box sx={{ flexGrow: 1 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                          <Typography variant="body2" sx={{ color: '#b8c5d6', fontSize: '0.8rem' }}>
                            CPU
                          </Typography>
                          <Typography variant="body2" sx={{ color: 'white', fontSize: '0.8rem' }}>
                            {agent.cpu}%
                          </Typography>
                        </Box>
                        <Box sx={{ 
                          height: 4, 
                          background: 'rgba(255, 255, 255, 0.1)', 
                          borderRadius: 2,
                          overflow: 'hidden'
                        }}>
                          <Box sx={{ 
                            width: `${agent.cpu}%`, 
                            height: '100%', 
                            background: agent.cpu > 80 ? '#ff6b6b' : agent.cpu > 60 ? '#ffa726' : '#4ecdc4',
                            transition: 'width 0.3s ease'
                          }} />
                        </Box>
                      </Box>
                    </Box>
                  </Box>

                  {/* Capabilities */}
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" sx={{ color: 'white', mb: 2, fontWeight: 600 }}>
                      Capabilities
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                      {agent.capabilities.map((capability) => (
                        <Chip
                          key={capability}
                          label={capability}
                          size="small"
                          sx={{
                            background: 'rgba(102, 126, 234, 0.2)',
                            color: '#667eea',
                            fontSize: '0.7rem'
                          }}
                        />
                      ))}
                    </Box>
                  </Box>

                  {/* Last Activity */}
                  <Box sx={{ 
                    p: 2, 
                    borderRadius: '8px', 
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid rgba(255, 255, 255, 0.1)'
                  }}>
                    <Typography variant="subtitle2" sx={{ color: 'white', mb: 1, fontWeight: 600 }}>
                      Last Activity
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#b8c5d6', fontSize: '0.85rem' }}>
                      {agent.lastActivity}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        ))}
      </Grid>
    </Box>
  )
}

export default Agents